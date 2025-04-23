import os
from tkinter import NONE
from lightning.pytorch.utilities.model_summary import summarize
from gluonts.evaluation import MultivariateEvaluator, make_evaluation_predictions
from gluonts.model.forecast_generator import DistributionForecastGenerator, SampleForecastGenerator
from gluonts.time_feature._base import second_of_minute, minute_of_hour, hour_of_day, day_of_year
from gluonts.transform import ExpectedNumInstanceSampler, ValidationSplitSampler
import logging
import torch
import gc
import time # Added for load_study retry delay
import inspect
from itertools import product
import collections.abc
from pathlib import Path
import subprocess
# Imports for Optuna
import optuna
from optuna import create_study, load_study
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner, MedianPruner, PercentilePruner, NopPruner
from optuna_integration import PyTorchLightningPruningCallback

import lightning.pytorch as pl # Import pl alias
from optuna.trial import TrialState # Added for checking trial status
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from optuna import create_study
from mysql.connector import connect as sql_connect
from optuna.storages import JournalStorage, RDBStorage
from optuna.storages.journal import JournalFileBackend

from wind_forecasting.utils.optuna_visualization import launch_optuna_dashboard, log_optuna_visualizations_to_wandb
from wind_forecasting.utils.optuna_table import log_detailed_trials_table_to_wandb
# from wind_forecasting.utils.trial_utils import handle_trial_with_oom_protection

import random
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_df_setup_params(model, model_config):
    study_name = f"tuning_{model}_{model_config['experiment']['run_name']}"
    optuna_cfg = model_config["optuna"]
    storage_cfg = optuna_cfg.get("storage", {})
    logging_cfg = model_config["logging"]
    experiment_cfg = model_config["experiment"]

    # Resolve paths relative to project root and substitute known variables
    project_root = experiment_cfg.get("project_root", os.getcwd())
    
    # Resolve paths with direct substitution
    optuna_dir_from_config = logging_cfg.get("optuna_dir")
    resolved_optuna_dir = resolve_path(project_root, optuna_dir_from_config)
    if not resolved_optuna_dir:
        raise ValueError("logging.optuna_dir is required but not found or resolved.")

    pgdata_path_from_config = storage_cfg.get("pgdata_path")
    resolved_pgdata_path = resolve_path(project_root, pgdata_path_from_config)

    socket_dir_base_from_config = storage_cfg.get("socket_dir_base")
    if not socket_dir_base_from_config:
        socket_dir_base_str = os.path.join(resolved_optuna_dir, "sockets")
    else:
        socket_dir_base_str = str(socket_dir_base_from_config).replace("${logging.optuna_dir}", resolved_optuna_dir)
    resolved_socket_dir_base = resolve_path(project_root, socket_dir_base_str) # Make absolute

    sync_dir_from_config = storage_cfg.get("sync_dir")
    if not sync_dir_from_config:
        # Default value uses the resolved optuna_dir
        sync_dir_str = os.path.join(resolved_optuna_dir, "sync")
    else:
        # Substitute directly if the variable exists
        sync_dir_str = str(sync_dir_from_config).replace("${logging.optuna_dir}", resolved_optuna_dir)
    resolved_sync_dir = resolve_path(project_root, sync_dir_str) # Make absolute

    db_setup_params = {
        "backend": storage_cfg.get("backend", "sqlite"),
        "project_root": project_root,
        "pgdata_path": resolved_pgdata_path,
        "study_name": study_name,
        "use_socket": storage_cfg.get("use_socket", True),
        "use_tcp": storage_cfg.get("use_tcp", False),
        "db_host": storage_cfg.get("db_host", "localhost"),
        "db_port": storage_cfg.get("db_port", 5432),
        "db_name": storage_cfg.get("db_name", "optuna_study_db"),
        "db_user": storage_cfg.get("db_user", "optuna_user"),
        "run_cmd_shell": storage_cfg.get("run_cmd_shell", False),
        "socket_dir_base": resolved_socket_dir_base,
        "sync_dir": resolved_sync_dir,
        "storage_dir": resolved_optuna_dir, # For non-postgres backends
        "sqlite_path": storage_cfg.get("sqlite_path"), # For sqlite
        "sqlite_wal": storage_cfg.get("sqlite_wal", True), # For sqlite
        "sqlite_timeout": storage_cfg.get("sqlite_timeout", 600), # For sqlite
    }
    return db_setup_params


# make paths absolute
def resolve_path(base_path, path_input):
    if not path_input: return None
    # Convert potential Path object back to string if needed
    path_str = str(path_input)
    abs_path = Path(path_str)
    if not abs_path.is_absolute():
        abs_path = Path(base_path) / abs_path
    return str(abs_path.resolve())

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

# Wrapper class to safely pass the Optuna pruning callback to PyTorch Lightning
class SafePruningCallback(pl.Callback):
    def __init__(self, trial: optuna.trial.Trial, monitor: str):
        super().__init__()
        # Instantiate the actual Optuna callback internally
        self.optuna_pruning_callback = PyTorchLightningPruningCallback(trial, monitor)

    # Delegate the relevant callback method(s)
    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        logging.info(f"DEBUG: SafePruningCallback.on_validation_end called for trial {self.optuna_pruning_callback._trial.number}, step {trainer.global_step}")
        # Call the corresponding method on the wrapped Optuna callback
        self.optuna_pruning_callback.on_validation_end(trainer, pl_module)

    # Delegate check_pruned if needed
    def check_pruned(self) -> None:
        self.optuna_pruning_callback.check_pruned()

class MLTuningObjective:
    def __init__(self, *, model, config, lightning_module_class, estimator_class, 
                 distr_output_class, max_epochs, limit_train_batches, data_module, 
                 metric, seed=42, tuning_phase=0, resample_freq_choices=None):
        self.model = model
        self.config = config
        self.lightning_module_class = lightning_module_class
        self.estimator_class = estimator_class
        self.distr_output_class = distr_output_class
        self.data_module = data_module
        self.metric = metric # TODO unused
        self.evaluator = MultivariateEvaluator(num_workers=None, custom_eval_fn=None)
        self.metrics = []
        self.seed = seed
        self.tuning_phase = tuning_phase
        self.resample_freq_choices = resample_freq_choices

        self.config["trainer"]["max_epochs"] = max_epochs
        self.config["trainer"]["limit_train_batches"] = limit_train_batches

        # Store pruning configuration
        self.pruning_enabled = "pruning" in config["optuna"] and config["optuna"]["pruning"].get("enabled", False)
        if self.pruning_enabled:
            logging.info(f"Pruning is enabled using {config['optuna']['pruning'].get('type', 'hyperband')} pruner")
        else:
            logging.info("Pruning is disabled")

        # Add GPU monitoring
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.device = torch.cuda.current_device()
            self.gpu_name = torch.cuda.get_device_name(self.device)
            self.total_memory = torch.cuda.get_device_properties(self.device).total_memory
            logging.info(f"GPU monitoring initialized for {self.gpu_name}")

    def log_gpu_stats(self, stage=""):
        """Log GPU memory usage at different stages of training"""
        if not self.gpu_available:
            return

        # Memory in GB
        allocated = torch.cuda.memory_allocated(self.device) / 1e9
        reserved = torch.cuda.memory_reserved(self.device) / 1e9
        max_allocated = torch.cuda.max_memory_allocated(self.device) / 1e9
        total = self.total_memory / 1e9

        # Calculate utilization percentage
        utilization_percent = (allocated / total) * 100

        logging.info(f"GPU Stats {stage}: "
                    f"Current Memory: {allocated:.2f}GB ({utilization_percent:.1f}%), "
                    f"Reserved: {reserved:.2f}GB, "
                    f"Peak: {max_allocated:.2f}GB, "
                    f"Total: {total:.2f}GB")

    def __call__(self, trial):
        # Set random seeds for reproducibility within each trial
        # Use different but deterministic seeds for each trial by combining base seed with trial number
        trial_seed = self.seed + trial.number
        torch.manual_seed(trial_seed)
        torch.cuda.manual_seed_all(trial_seed)

        random.seed(trial_seed)
        np.random.seed(trial_seed)
        logging.info(f"Set random seed for trial {trial.number} to {trial_seed}")

        # Initialize wandb logger for this trial only on rank 0
        wandb_logger_trial = None # Initialize to None for non-rank-0 workers

        # Log GPU stats at the beginning of the trial
        self.log_gpu_stats(stage=f"Trial {trial.number} Start")
      
        params = self.estimator_class.get_params(trial, self.tuning_phase, 
                                                 dynamic_kwargs={"resample_freq": self.resample_freq_choices})
        if "resample_freq" in params or "per_turbine" in params:
            self.data_module.freq = f"{params['resample_freq']}s"
            self.data_module.per_turbine = params["per_turbine"]
            self.data_module.set_train_ready_path()
            assert os.path.exists(self.data_module.train_ready_data_path), "Must generates dataset and splits in tuning.py, rank 0. Requested resampling frequency may not be compatible."
            self.data_module.generate_splits(save=True, reload=False, splits=["train", "val"])

        estimator_sig = inspect.signature(self.estimator_class.__init__)
        estimator_params = [param.name for param in estimator_sig.parameters.values()]

        if "dim_feedforward" not in params and "d_model" in params:
            # set dim_feedforward to 4x the d_model found in this trial
            params["dim_feedforward"] = params["d_model"] * 4
        elif "d_model" in estimator_params and estimator_sig.parameters["d_model"].default is not inspect.Parameter.empty:
            # if d_model is not contained in the trial but is a paramter, get the default
            params["dim_feedforward"] = estimator_sig.parameters["d_model"].default * 4

        logging.info(f"Testing params {tuple((k, v) for k, v in params.items())}")

        self.config["model"]["distr_output"]["kwargs"].update({k: v for k, v in params.items() if k in self.config["model"]["distr_output"]["kwargs"]})
        self.config["dataset"].update({k: v for k, v in params.items() if k in self.config["dataset"]})
        self.config["model"][self.model].update({k: v for k, v in params.items() if k in self.config["model"][self.model]})
        self.config["trainer"].update({k: v for k, v in params.items() if k in self.config["trainer"]})


        trial_callbacks = [] # Start with an empty list for this trial

        # Add pruning callback if enabled
        trial_pruning_callback = None # Initialize
        if self.pruning_enabled:
            pruning_monitor_metric = self.config.get("trainer", {}).get("monitor_metric", "val_loss")
            trial_pruning_callback = SafePruningCallback(trial, monitor=pruning_monitor_metric)
            trial_callbacks.append(trial_pruning_callback)
            logging.info(f"Added pruning callback for trial {trial.number}, monitoring '{pruning_monitor_metric}'")

        # Instantiate trial-specific ModelCheckpoint
        checkpoint_config = self.config['callbacks']['model_checkpoint']['init_args']
        filename_template = checkpoint_config.get('filename', '{epoch}-{step}-{val_loss:.2f}')
        # Ensure unique filename per trial
        trial_filename = f"trial_{trial.number}_" + filename_template
        
        # Resolve checkpoint directory path
        resolved_checkpoint_dir = resolve_path(
            self.config['experiment']['project_root'], 
            checkpoint_config.get('dirpath', self.config['logging']['checkpoint_dir'])
        )
        if not resolved_checkpoint_dir:
             raise ValueError("Could not resolve checkpoint directory path.")

        trial_checkpoint_callback = ModelCheckpoint(
            dirpath=resolved_checkpoint_dir,
            filename=trial_filename,
            monitor=checkpoint_config.get('monitor', 'val_loss'),
            mode=checkpoint_config.get('mode', 'min'),
            save_top_k=checkpoint_config.get('save_top_k', 1),
            save_last=checkpoint_config.get('save_last', True) # Keep save_last if needed
            # Add any other relevant args from checkpoint_config here if necessary
        )
        trial_callbacks.append(trial_checkpoint_callback)
        logging.info(f"Trial {trial.number}: Created specific ModelCheckpoint: {trial_checkpoint_callback} saving to {resolved_checkpoint_dir}/{trial_filename}")

        # Add other callbacks from config, EXCLUDING ModelCheckpoint
        instantiated_callbacks_from_config = self.config.get("callbacks", [])
        if not isinstance(instantiated_callbacks_from_config, list) or not all(isinstance(cb, pl.Callback) for cb in instantiated_callbacks_from_config):
             logging.warning("Callbacks in config are not a list of instantiated pl.Callback objects. Ignoring them for this trial.")
             instantiated_callbacks_from_config = []
        
        for cb in instantiated_callbacks_from_config:
            if not isinstance(cb, ModelCheckpoint):
                trial_callbacks.append(cb)
            else:
                logging.info(f"Trial {trial.number}: Skipping pre-instantiated ModelCheckpoint from config.")

        logging.info(f"Trial {trial.number}: Final callbacks list: {[type(cb).__name__ for cb in trial_callbacks]}")
        # --- End Callback Preparation ---

        # Prepare trainer_kwargs, ensuring it uses the trial-specific callbacks
        trial_trainer_kwargs = {k: v for k, v in self.config["trainer"].items() if k != 'callbacks'}
        trial_trainer_kwargs["callbacks"] = trial_callbacks
        logging.debug(f"Final callbacks passed to estimator: {[type(cb).__name__ for cb in trial_trainer_kwargs['callbacks']]}")

        # Remove monitor_metric if it exists, as it's handled by ModelCheckpoint
        if "monitor_metric" in trial_trainer_kwargs:
            del trial_trainer_kwargs["monitor_metric"]

        # Initialize W&B for ALL workers
        try:
            # Construct unique run name and tags
            run_name = f"{self.config['experiment']['run_name']}_rank_{os.environ.get('WORKER_RANK', '0')}_trial_{trial.number}"

            # Clean and flatten the parameters for logging
            cleaned_params = {}
            model_prefix = f"{self.model}."
            config_prefix = "model_config."

            for k, v in trial.params.items():
                cleanedKeyCandidate = k
                if cleanedKeyCandidate.startswith(model_prefix):
                    cleanedKeyCandidate = cleanedKeyCandidate[len(model_prefix):]

                if cleanedKeyCandidate.startswith(config_prefix):
                    cleaned_key = cleanedKeyCandidate[len(config_prefix):]
                else:
                    cleaned_key = cleanedKeyCandidate

                # Store the cleaned key and value
                cleaned_params[cleaned_key] = v

            cleaned_params["optuna_trial_number"] = trial.number

            # Initialize a new W&B run for this specific trial
            wandb.init(
                # Core identification
                project=self.config['experiment'].get('project_name', 'wind_forecasting'),
                entity=self.config['logging'].get('entity', None),
                group=self.config['experiment']['run_name'],
                name=run_name,
                job_type="optuna_trial",
                dir=self.config['logging']['wandb_dir'],
                # Configuration and Metadata
                config=cleaned_params, # Use the cleaned dictionary
                tags=[self.model] + self.config['experiment'].get('extra_tags', []),
                notes=f"Optuna trial {trial.number} (Rank {os.environ.get('WORKER_RANK', '0')}) for study: {self.config['experiment'].get('notes', '')}",
                # Logging and Behavior
                save_code=self.config['optuna'].get('save_trial_code', False),
                mode=self.config['logging'].get('wandb_mode', 'online'),
                reinit=True
            )
            logging.info(f"Rank {os.environ.get('WORKER_RANK', '0')}: Initialized W&B run '{run_name}' for trial {trial.number}")

            # Create a WandbLogger using the current W&B run
            # log_model=False as we only want metrics for this trial logger
            wandb_logger_trial = WandbLogger(log_model=False, experiment=wandb.run)
            logging.info(f"Rank {os.environ.get('WORKER_RANK', '0')}: Created WandbLogger for trial {trial.number}")

            # Add the trial-specific logger to the trainer kwargs for this worker
            trial_trainer_kwargs["logger"] = wandb_logger_trial
            logging.info(f"Rank {os.environ.get('WORKER_RANK', '0')}: Added trial-specific WandbLogger to trainer_kwargs for trial {trial.number}")

        except Exception as e:
            logging.error(f"Rank {os.environ.get('WORKER_RANK', '0')}: Failed to initialize W&B or create logger for trial {trial.number}: {e}", exc_info=True)
            # Ensure wandb_logger_trial remains None if setup fails
            wandb_logger_trial = None

        # Verify GPU configuration before creating estimator
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            logging.info(f"Creating estimator using GPU {device}: {torch.cuda.get_device_name(device)}")

            # Ensure we have the right GPU configuration in trainer_kwargs
            if "devices" in self.config["trainer"] and self.config["trainer"]["devices"] > 1:
                if "CUDA_VISIBLE_DEVICES" in os.environ and len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) == 1:
                    logging.warning(f"Overriding trainer devices={self.config['trainer']['devices']} to 1 due to CUDA_VISIBLE_DEVICES")
                    self.config["trainer"]["devices"] = 1
                    self.config["trainer"]["strategy"] = "auto"
        else:
            logging.warning("No CUDA available for estimator creation")

        context_length_factor = params.get('context_length_factor', self.config["dataset"].get("context_length_factor", 2)) # Default to config or 2 if not in trial/config
        context_length = int(context_length_factor * self.data_module.prediction_length)

        # Estimator Arguments to handle difference between models
        estimator_kwargs = {
            "freq": self.data_module.freq,
            "prediction_length": self.data_module.prediction_length,
            "context_length": context_length,
            "num_feat_dynamic_real": self.data_module.num_feat_dynamic_real,
            "num_feat_static_cat": self.data_module.num_feat_static_cat,
            "cardinality": self.data_module.cardinality,
            "num_feat_static_real": self.data_module.num_feat_static_real,
            "input_size": self.data_module.num_target_vars,
            "scaling": False,
            "lags_seq": [0],
            "use_lazyframe": False,
            "batch_size": self.config["dataset"].get("batch_size", 128),
            "num_batches_per_epoch": trial_trainer_kwargs["limit_train_batches"], # Use value from trial_trainer_kwargs
            "train_sampler": ExpectedNumInstanceSampler(num_instances=1.0, min_past=context_length, min_future=self.data_module.prediction_length), # Use calculated context_length
            "validation_sampler": ValidationSplitSampler(min_past=context_length, min_future=self.data_module.prediction_length), # Use calculated context_length
            "time_features": [second_of_minute, minute_of_hour, hour_of_day, day_of_year],
            # Include distr_output initially, will be removed conditionally
            "distr_output": self.distr_output_class(dim=self.data_module.num_target_vars, **self.config["model"]["distr_output"]["kwargs"]),
            "trainer_kwargs": trial_trainer_kwargs, # Pass the trial-specific kwargs
            "num_parallel_samples": self.config["model"][self.model].get("num_parallel_samples", 100) if self.model == 'tactis' else 100, # Default 100 if not specified
        }
        # Add model-specific arguments from the default config YAML
        estimator_kwargs.update(self.config["model"][self.model])
        
        # Add model-specific tunable hyperparameters suggested by Optuna trial
        valid_estimator_params = set(estimator_params)
        filtered_params = {
            k: v for k, v in params.items()
            if k in valid_estimator_params and k != 'context_length_factor'
        }
        logging.info(f"Trial {trial.number}: Updating estimator_kwargs with filtered params: {list(filtered_params.keys())}")
        estimator_kwargs.update(filtered_params)

        # TACTiS manages its own distribution output internally, remove if present
        if self.model == 'tactis' and 'distr_output' in estimator_kwargs:
            estimator_kwargs.pop('distr_output')

        logging.info(f"Trial {trial.number}: Instantiating estimator '{self.model}' with final args: {list(estimator_kwargs.keys())}")
        

        agg_metrics = None
        train_output = None # Initialize train_output
        best_model_path_for_eval = None # Initialize path variable
        best_score_for_return = None # Initialize score variable

        try:
            try:
                estimator = self.estimator_class(**estimator_kwargs)
            except Exception as e:
                logging.error(f"Trial {trial.number} - Error creating estimator: {str(e)}", exc_info=True)
                raise optuna.exceptions.TrialPruned(f"Error creating estimator in trial {trial.number}: {str(e)}")

            # Log GPU stats before training
            self.log_gpu_stats(stage=f"Trial {trial.number} Before Training")

            # Conditionally Create Forecast Generator
            try:
                if self.model == 'tactis':
                    logging.info(f"Trial {trial.number}: Using SampleForecastGenerator for TACTiS model.")
                    forecast_generator = SampleForecastGenerator()
                else:
                    logging.info(f"Trial {trial.number}: Using DistributionForecastGenerator for {self.model} model.")
                    if not hasattr(estimator, 'distr_output'):
                         raise AttributeError(f"Estimator for model '{self.model}' is missing 'distr_output' attribute needed for DistributionForecastGenerator.")
                    forecast_generator = DistributionForecastGenerator(estimator.distr_output)
            except Exception as e:
                logging.error(f"Trial {trial.number} - Error creating forecast generator: {str(e)}", exc_info=True)
                raise optuna.exceptions.TrialPruned(f"Error creating forecast generator in trial {trial.number}: {str(e)}")

            # Log trainer state before attempting training
            try:
                trainer_state_str = str(estimator.trainer.state) if hasattr(estimator, 'trainer') and hasattr(estimator.trainer, 'state') else 'N/A'
                trainer_epoch_str = str(estimator.trainer.current_epoch) if hasattr(estimator, 'trainer') and hasattr(estimator.trainer, 'current_epoch') else 'N/A'
                logging.info(f"Trial {trial.number}: Trainer state before train: {trainer_state_str}, Current epoch: {trainer_epoch_str}")
            except Exception as log_e:
                logging.warning(f"Trial {trial.number}: Could not log trainer state before train: {log_e}")

            try:
                # Call estimator.train without the ckpt_path argument, as ModelCheckpoint is handled via callbacks
                logging.info(f"Trial {trial.number}: Starting estimator.train()...")
                train_output = estimator.train(
                    training_data=self.data_module.train_dataset,
                    validation_data=self.data_module.val_dataset,
                    forecast_generator=forecast_generator
                )
                logging.info(f"Trial {trial.number}: Finished estimator.train() successfully.")

            except optuna.exceptions.TrialPruned as e:
                # If Optuna pruning callback raised TrialPruned, log it and re-raise
                logging.warning(f"Trial {trial.number} - Pruned during training by Optuna callback: {str(e)}", exc_info=True)
                raise e # Re-raise the original TrialPruned exception
            except Exception as e:
                # Catch any other exception during training
                logging.error(f"Trial {trial.number} - Error during model training: {str(e)}", exc_info=True)
                # Raise a *new* TrialPruned exception to signal Optuna
                raise optuna.exceptions.TrialPruned(f"Error during model training in trial {trial.number}: {str(e)}")

            # Log GPU stats after training
            self.log_gpu_stats(stage=f"Trial {trial.number} After Training")
            
            # --- Retrieve Best Path and Score Directly from Trial's Checkpoint Callback ---
            if trial_checkpoint_callback:
                best_model_path_for_eval = trial_checkpoint_callback.best_model_path
                best_score_for_return = trial_checkpoint_callback.best_model_score
                logging.info(f"Trial {trial.number}: Retrieved directly from trial's ModelCheckpoint - Path: {best_model_path_for_eval}, Score: {best_score_for_return}")
            else:
                logging.error(f"Trial {trial.number}: Trial-specific ModelCheckpoint callback not found after training. Cannot retrieve best path/score.")
                raise optuna.exceptions.TrialPruned(f"Trial {trial.number} failed: Checkpoint callback missing post-training.")

            if not best_model_path_for_eval:
                 logging.warning(f"Trial {trial.number}: best_model_path from trial's ModelCheckpoint is empty or None. Using last_model_path if available.")
                 if hasattr(trial_checkpoint_callback, 'last_model_path') and trial_checkpoint_callback.last_model_path:
                      best_model_path_for_eval = trial_checkpoint_callback.last_model_path
                      logging.info(f"Trial {trial.number}: Using last_model_path for evaluation: {best_model_path_for_eval}")
                 else:
                      logging.error(f"Trial {trial.number}: No best or last checkpoint path found in trial's ModelCheckpoint. Cannot proceed with evaluation.")
                      raise optuna.exceptions.TrialPruned(f"Trial {trial.number} failed: No checkpoint path found.")

            # --- Evaluation Logic (Using retrieved best_model_path_for_eval) ---
            predictor = None # Initialize predictor
            try:
                loaded_module = self.lightning_module_class.load_from_checkpoint(
                    best_model_path_for_eval, # Use the path retrieved directly
                )
                logging.info(f"Trial {trial.number}: Successfully loaded LightningModule state from {best_model_path_for_eval} for evaluation.")
                transformation = estimator.create_transformation()
                logging.info(f"Trial {trial.number}: Created transformation pipeline for evaluation.")
                predictor = estimator.create_predictor(
                    transformation=transformation,
                    module=loaded_module
                )
                logging.info(f"Trial {trial.number}: Successfully created predictor using loaded module state for evaluation.")
            except FileNotFoundError:
                 logging.error(f"Trial {trial.number}: Checkpoint file not found at {best_model_path_for_eval}. Cannot load predictor for evaluation.", exc_info=True)
                 raise optuna.exceptions.TrialPruned(f"Checkpoint file not found for evaluation in trial {trial.number}")
            except Exception as e_load:
                logging.error(f"Trial {trial.number}: Failed to load module or create predictor from checkpoint {best_model_path_for_eval} for evaluation: {e_load}", exc_info=True)
                raise optuna.exceptions.TrialPruned(f"Failed to load predictor from checkpoint for evaluation in trial {trial.number}")

            if predictor is None:
                 logging.error(f"Trial {trial.number}: Predictor is None after attempting to load from checkpoint. Cannot proceed with evaluation.")
                 raise optuna.exceptions.TrialPruned(f"Predictor loading failed unexpectedly for evaluation in trial {trial.number}")

            # Calculate evaluation metrics (agg_metrics) for logging purposes
            try:
                eval_kwargs = {
                    "dataset": self.data_module.val_dataset,
                    "predictor": predictor,
                }
                if self.model == 'tactis':
                    logging.info(f"Trial {trial.number}: Evaluating TACTiS using SampleForecast for logging.")
                else:
                    eval_kwargs["output_distr_params"] = {"loc": "mean", "cov_factor": "cov_factor", "cov_diag": "cov_diag"}
                    logging.info(f"Trial {trial.number}: Evaluating {self.model} using DistributionForecast for logging with params: {eval_kwargs['output_distr_params']}")

                forecast_it, ts_it = make_evaluation_predictions(**eval_kwargs)
            except Exception as e:
                logging.error(f"Trial {trial.number} - Error making evaluation predictions for logging: {str(e)}", exc_info=True)
                raise optuna.exceptions.TrialPruned(f"Error making evaluation predictions for logging in trial {trial.number}: {str(e)}")

            try:
                agg_metrics, _ = self.evaluator(ts_it, forecast_it, num_series=self.data_module.num_target_vars)
                
                # Calculate trainable parameters using the original estimator's method
                try:
                     trainable_params = summarize(estimator.create_lightning_module()).trainable_parameters
                except Exception as summary_err:
                     logging.warning(f"Trial {trial.number}: Could not get trainable parameters via summarize: {summary_err}. Setting to 0.")
                     trainable_params = 0
                agg_metrics["trainable_parameters"] = trainable_params

                self.metrics.append(agg_metrics.copy()) # Keep track if needed elsewhere
                
                # Log available metrics for debugging
                logging.info(f"Trial {trial.number} - Aggregated metrics calculated for logging: {list(agg_metrics.keys())}")
            except Exception as e:
                logging.error(f"Trial {trial.number} - Error computing evaluation metrics for logging: {str(e)}", exc_info=True)
                raise optuna.exceptions.TrialPruned(f"Error computing evaluation metrics for logging in trial {trial.number}: {str(e)}")
            # --- End of Evaluation Logic ---

            # Log GPU stats at the end of the trial
            self.log_gpu_stats(stage=f"Trial {trial.number} End")

            # Force garbage collection at the end of each trial
            gc.collect()
            torch.cuda.empty_cache()

        finally:
            # Check if wandb_logger_trial was created (only on rank 0) and if wandb.run is active
            if wandb_logger_trial is not None and wandb.run is not None:
                logging.info(f"Rank 0: Finishing trial-specific W&B run '{wandb.run.name}' for trial {trial.number}")
                wandb.finish()
            elif os.environ.get('WORKER_RANK', '0') == '0' and wandb.run is not None:
                # If logger wasn't assigned but a run exists on rank 0, try finishing it.
                logging.warning(f"Rank 0: wandb_logger_trial was None, but an active W&B run ('{wandb.run.name}') was found. Attempting to finish.")
                wandb.finish()
            # Explicit cleanup attempt between trials
            logging.info(f"Trial {trial.number} - Attempting cleanup...")
            if torch.cuda.is_available():
                logging.info(f"Trial {trial.number} - Clearing CUDA cache.")
                torch.cuda.empty_cache()
            logging.info(f"Trial {trial.number} - Triggering garbage collection.")
            gc.collect()
            logging.info(f"Trial {trial.number} - Cleanup attempt finished.")

        # --- Return Best Metric Logic (Using retrieved best_score_for_return) ---
        metric_to_return = self.config.get("trainer", {}).get("monitor_metric", "val_loss")
        metric_value = None

        try:
            if best_score_for_return is None:
                # This might happen if no validation epoch completed successfully or the metric wasn't logged
                error_msg = f"'best_model_score' from trial's ModelCheckpoint for '{metric_to_return}' is None."
                logging.error(f"Trial {trial.number} - {error_msg}")
                raise optuna.exceptions.TrialPruned(f"Trial {trial.number} failed: {error_msg}")

            # Convert to float (handles tensors, numpy arrays, etc.)
            if hasattr(best_score_for_return, 'item'):
                metric_value = float(best_score_for_return.item())
            elif isinstance(best_score_for_return, (np.ndarray, torch.Tensor)) and best_score_for_return.size == 1:
                metric_value = float(best_score_for_return.item())
            else:
                metric_value = float(best_score_for_return) # Try direct conversion

            logging.info(f"Trial {trial.number} - Returning best score '{metric_to_return}' from trial's ModelCheckpoint to Optuna: {metric_value}")
            return metric_value

        except (TypeError, ValueError) as e:
             error_msg = f"Error converting best_model_score (value: {best_score_for_return}) for metric '{metric_to_return}' to float: {e}"
             logging.error(f"Trial {trial.number} - {error_msg}", exc_info=True)
             raise optuna.exceptions.TrialPruned(f"Trial {trial.number} failed: {error_msg}")
        except optuna.exceptions.TrialPruned as e: # Re-raise TrialPruned explicitly
             raise e
        except Exception as e:
            error_msg = f"Unexpected error retrieving or processing best_model_score for metric '{metric_to_return}': {e}"
            logging.error(f"Trial {trial.number} - {error_msg}", exc_info=True)
            raise optuna.exceptions.TrialPruned(f"Trial {trial.number} failed: {error_msg}")


def get_tuned_params(storage, study_name):
    logging.info(f"Getting storage for Optuna study {study_name}.")  
    try:
        study_id = storage.get_study_id_from_name(study_name)
    except Exception:
        raise FileNotFoundError(f"Optuna study {study_name} not found. Please run tune_hyperparameters_multi for all outputs first.")
    # self.model[output].set_params(**storage.get_best_trial(study_id).params)
    # storage.get_all_studies()[0]._study_id
    # estimato = self.create_model(**storage.get_best_trial(study_id).params)
    return storage.get_best_trial(study_id).params

# Update signature: Add optuna_storage_url, remove storage_dir, use_rdb, restart_study
def tune_model(model, config, study_name, optuna_storage, lightning_module_class, estimator_class,
               max_epochs, limit_train_batches,
               distr_output_class, data_module,
               metric="val_loss", direction="minimize", n_trials_per_worker=10,
               trial_protection_callback=None, seed=42, tuning_phase=0): # Removed wandb_run_id

    # Log safely without credentials if they were included (they aren't for socket trust)
    if hasattr(optuna_storage, "url"):
        log_storage_url = optuna_storage.url.split('@')[0] + '@...' if '@' in optuna_storage.url else optuna_storage.url
        logging.info(f"Using Optuna storage URL: {log_storage_url}")

    # NOTE: Restarting the study is now handled in the Slurm script by deleting the PGDATA directory

    # Configure pruner based on settings
    pruner = None
    if "pruning" in config["optuna"] and config["optuna"]["pruning"].get("enabled", False):
        pruning_type = config["optuna"]["pruning"].get("type", "hyperband").lower()
        min_resource = config["optuna"]["pruning"].get("min_resource", 2)
        max_resource = config["optuna"]["pruning"].get("max_resource", max_epochs)

        logging.info(f"Configuring pruner: type={pruning_type}, min_resource={min_resource}, max_resource={max_resource}")

        if pruning_type == "hyperband":
            reduction_factor = config["optuna"]["pruning"].get("reduction_factor", 2)
            pruner = HyperbandPruner(
                min_resource=min_resource,
                max_resource=max_resource,
                reduction_factor=reduction_factor
            )
            logging.info(f"Created HyperbandPruner with min_resource={min_resource}, max_resource={max_resource}, reduction_factor={reduction_factor}")
        elif pruning_type == "median":
            n_warmup_steps = config["optuna"]["pruning"].get("n_warmup_steps", 2)
            if "n_warmup_steps" not in config["optuna"]["pruning"]:
                logging.warning(f"YAML config missing 'optuna.pruning.n_warmup_steps', defaulting to {n_warmup_steps}")

            n_startup_trials = config["optuna"]["pruning"].get("n_startup_trials", 5)  # Default to 5 if missing
            if "n_startup_trials" not in config["optuna"]["pruning"]:
                logging.warning(f"YAML config missing 'optuna.pruning.n_startup_trials', defaulting to {n_startup_trials}")

            pruner = MedianPruner(
                n_startup_trials=n_startup_trials,
                n_warmup_steps=n_warmup_steps
            )
            logging.info(f"Created MedianPruner with n_startup_trials={n_startup_trials}, n_warmup_steps={n_warmup_steps}")
        elif pruning_type == "percentile":
            percentile = config["optuna"]["pruning"].get("percentile", 25)
            pruner = PercentilePruner(percentile=percentile, n_startup_trials=5, n_warmup_steps=min_resource)
            logging.info(f"Created PercentilePruner with percentile={percentile}, n_startup_trials=5, n_warmup_steps={min_resource}")
        else:
            logging.warning(f"Unknown pruner type: {pruning_type}, using no pruning")
            pruner = NopPruner()
    else:
        logging.info("Pruning is disabled, using NopPruner")
        pruner = NopPruner()

    # Get worker ID for study creation/loading logic
    # Use WORKER_RANK consistent with run_model.py. Default to '0' if not set.
    worker_id = os.environ.get('WORKER_RANK', '0')

    # Create study on rank 0, load on other ranks
    study = None # Initialize study variable
    try:
        if worker_id == '0':
            logging.info(f"Rank 0: Creating/loading Optuna study '{study_name}' with pruner: {type(pruner).__name__}")
            study = create_study(
                study_name=study_name,
                storage=optuna_storage,
                direction=direction,
                load_if_exists=True, # Rank 0 handles creation or loading
                sampler=TPESampler(seed=seed),
                pruner=pruner
            )
            logging.info(f"Rank 0: Study '{study_name}' created or loaded successfully.")

            # --- Launch Dashboard (Rank 0 only) ---
            if hasattr(optuna_storage, "url"):
                launch_optuna_dashboard(config, optuna_storage.url) # Call imported function
            # --------------------------------------
        else:
            # Non-rank-0 workers MUST load the study created by rank 0
            logging.info(f"Rank {worker_id}: Attempting to load existing Optuna study '{study_name}'")
            # Add a small delay and retry mechanism for loading, in case rank 0 is slightly delayed
            max_retries = 6 # Increased retries slightly
            retry_delay = 10 # Increased delay slightly
            for attempt in range(max_retries):
                try:
                    study = load_study(
                        study_name=study_name,
                        storage=optuna_storage,
                        sampler=TPESampler(seed=seed), # Sampler might be needed for load_study too
                        pruner=pruner
                    )
                    logging.info(f"Rank {worker_id}: Study '{study_name}' loaded successfully on attempt {attempt+1}.")
                    break # Exit loop on success
                except KeyError as e: # Optuna <3.0 raises KeyError if study doesn't exist yet
                     if attempt < max_retries - 1:
                          logging.warning(f"Rank {worker_id}: Study '{study_name}' not found yet (attempt {attempt+1}/{max_retries}). Retrying in {retry_delay}s... Error: {e}")
                          time.sleep(retry_delay)
                     else:
                          logging.error(f"Rank {worker_id}: Failed to load study '{study_name}' after {max_retries} attempts (KeyError). Aborting.")
                          raise
                except Exception as e: # Catch other potential loading errors (e.g., DB connection issues)
                     logging.error(f"Rank {worker_id}: An unexpected error occurred while loading study '{study_name}' on attempt {attempt+1}: {e}", exc_info=True)
                     # Decide whether to retry on other errors or raise immediately
                     if attempt < max_retries - 1:
                          logging.warning(f"Retrying in {retry_delay}s...")
                          time.sleep(retry_delay)
                     else:
                          logging.error(f"Rank {worker_id}: Failed to load study '{study_name}' after {max_retries} attempts due to persistent errors. Aborting.")
                          raise # Re-raise other errors after retries

            # Check if study was successfully loaded after the loop
            if study is None:
                 # This condition should ideally be caught by the error handling within the loop, but added for safety.
                 raise RuntimeError(f"Rank {worker_id}: Could not load study '{study_name}' after multiple retries.")

    except Exception as e:
        # Log error with rank information
        logging.error(f"Rank {worker_id}: Error creating/loading study '{study_name}': {str(e)}", exc_info=True)
        # Log storage URL safely
        if hasattr(optuna_storage, "url"):
            log_storage_url_safe = str(optuna_storage.url).split('@')[0] + '@...' if '@' in str(optuna_storage.url) else str(optuna_storage.url)
            logging.error(f"Error details - Type: {type(e).__name__}, Storage: {log_storage_url_safe}")
        else:
            logging.error(f"Error details - Type: {type(e).__name__}, Storage: Journal")
        raise

    # Worker ID already fetched above for study creation/loading

    if tuning_phase == 0 and worker_id == "0":
        resample_freq_choices = config["optuna"]["resample_freq_choices"]
        per_turbine_choices = [True, False]
        for resample_freq, per_turbine in product(resample_freq_choices, per_turbine_choices):
            # for each combination of resample_freq and per_turbine, generate the datasets
            data_module.freq = f"{resample_freq}s"
            data_module.per_turbine_target = per_turbine
            data_module.set_train_ready_path()
            if not os.path.exists(data_module.train_ready_data_path):
                data_module.generate_datasets()
                reload = True
            else:
                reload = False
            data_module.generate_splits(save=True, reload=reload, splits=["train", "val"])

    logging.info(f"Worker {worker_id}: Participating in Optuna study {study_name}")

    # get from config
    resample_freq_choices = config.get("optuna", {}).get("resample_freq_choices", None)
    if resample_freq_choices is None:
        logging.warning("'optuna.resample_freq_choices' not found in config. Default to 60s.")
        resample_freq_choices = [60]

    tuning_objective = MLTuningObjective(model=model, config=config,
                                        lightning_module_class=lightning_module_class,
                                        estimator_class=estimator_class,
                                        distr_output_class=distr_output_class,
                                        max_epochs=max_epochs,
                                        limit_train_batches=limit_train_batches,
                                        data_module=data_module,
                                        metric=metric,
                                        seed=seed,
                                        tuning_phase=tuning_phase,
                                        resample_freq_choices=resample_freq_choices)

    # Use the trial protection callback if provided
    objective_fn = (lambda trial: trial_protection_callback(tuning_objective, trial)) if trial_protection_callback else tuning_objective

    # WandB integration deprecated
    optimize_callbacks = [] # Ensure optimize_callbacks is an empty list

    try:
        # Let Optuna handle trial distribution - each worker will ask the storage for a trial
        # Show progress bar only on rank 0 to avoid cluttered logs
        study.optimize(
            objective_fn,
            n_trials=n_trials_per_worker,
            callbacks=optimize_callbacks,
            show_progress_bar=(worker_id=='0')
        )
    except Exception as e:
        logging.error(f"Worker {worker_id}: Failed during study optimization: {str(e)}", exc_info=True)
        raise

    if worker_id == '0' and study:
        logging.info("Rank 0: Starting W&B summary run creation.")

        # Wait for all expected trials to complete
        num_workers = int(os.environ.get('WORLD_SIZE', 1))
        expected_total_trials = num_workers * n_trials_per_worker
        logging.info(f"Rank 0: Expecting a total of {expected_total_trials} trials ({num_workers} workers * {n_trials_per_worker} trials/worker).")

        logging.info("Rank 0: Waiting for all expected Optuna trials to reach a terminal state...")
        wait_interval_seconds = 30
        while True:
            # Refresh trials from storage
            all_trials_current = study.get_trials(deepcopy=False)
            finished_trials = [t for t in all_trials_current if t.state in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)]
            num_finished = len(finished_trials)
            num_total_in_db = len(all_trials_current) # Current count in DB

            logging.info(f"Rank 0: Trial status check: {num_finished} finished / {num_total_in_db} in DB (expected total: {expected_total_trials}).")

            if num_finished >= expected_total_trials:
                logging.info(f"Rank 0: All {expected_total_trials} expected trials have reached a terminal state.")
                break
            elif num_total_in_db > expected_total_trials and num_finished >= expected_total_trials:
                 logging.warning(f"Rank 0: Found {num_total_in_db} trials in DB (expected {expected_total_trials}), but {num_finished} finished trials meet the expectation.")
                 break

            logging.info(f"Rank 0: Still waiting for trials to finish ({num_finished}/{expected_total_trials}). Sleeping for {wait_interval_seconds} seconds...")
            time.sleep(wait_interval_seconds)

        try:
            # Fetch best trial *before* initializing summary run
            best_trial = None
            try:
                best_trial = study.best_trial
                logging.info(f"Rank 0: Fetched best trial: Number={best_trial.number}, Value={best_trial.value}")
            except ValueError:
                logging.warning("Rank 0: Could not retrieve best trial (likely no trials completed successfully).")
            except Exception as e_best_trial:
                logging.error(f"Rank 0: Error fetching best trial: {e_best_trial}", exc_info=True)

            # Fetch Git info directly using subprocess
            remote_url = None
            commit_hash = None
            try:
                # Get remote URL
                remote_url_bytes = subprocess.check_output(['git', 'config', '--get', 'remote.origin.url'], stderr=subprocess.STDOUT).strip()
                remote_url = remote_url_bytes.decode('utf-8')
                # Convert SSH URL to HTTPS URL if necessary
                if remote_url.startswith("git@"):
                    remote_url = remote_url.replace(":", "/").replace("git@", "https://")
                # Remove .git suffix AFTER potential conversion
                if remote_url.endswith(".git"):
                    remote_url = remote_url[:-4]

                # Get commit hash
                commit_hash_bytes = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.STDOUT).strip()
                commit_hash = commit_hash_bytes.decode('utf-8')
                logging.info(f"Rank 0: Fetched Git Info - URL: {remote_url}, Commit: {commit_hash}")
            except subprocess.CalledProcessError as e:
                logging.warning(f"Rank 0: Could not get Git info: {e.output.decode('utf-8').strip()}")
            except FileNotFoundError:
                logging.warning("Rank 0: 'git' command not found. Cannot log Git info.")
            except Exception as e_git:
                 logging.error(f"Rank 0: An unexpected error occurred while fetching Git info: {e_git}", exc_info=True)

            git_info_config = {}
            if remote_url and commit_hash:
                 git_info_config = {"git_info": {"url": remote_url, "commit": commit_hash}}
            else:
                 logging.warning("Rank 0: Git info could not be fully determined. Logging summary run without it.")


            # Determine summary run name
            base_run_name = config['experiment']['run_name']
            if best_trial:
                run_name = f"RESULTS_{base_run_name}_best_trial_{best_trial.number}"
            else:
                run_name = f"RESULTS_{base_run_name}_optuna_summary"

            project_name = config['experiment'].get('project_name', 'wind_forecasting')
            group_name = config['experiment']['run_name']
            wandb_dir = config['logging'].get('wandb_dir', './logging/wandb')
            tags = [model, "optuna_summary"] + config['experiment'].get('extra_tags', [])

            # Ensure wandb is not already initialized in a weird state (shouldn't be, but safety check)
            if wandb.run is not None:
                logging.warning(f"Rank 0: Found an existing W&B run ({wandb.run.id}) before starting summary run. Finishing it.")
                wandb.finish()

            wandb.init(
                name=run_name,
                project=project_name,
                group=group_name,
                job_type="optuna_summary",
                dir=wandb_dir,
                tags=tags,
                config=git_info_config,
                reinit=True # Allow reinitialization if needed, though the check above should handle most cases
            )
            logging.info(f"Rank 0: Initialized W&B summary run: {wandb.run.name} (ID: {wandb.run.id}) with Git info: {git_info_config}")

            try:
                # Log Optuna visualizations using the helper function
                logging.info("Rank 0: Logging Optuna visualizations to W&B summary run...")
                log_optuna_visualizations_to_wandb(study, wandb.run)
                logging.info("Rank 0: Finished logging Optuna visualizations to W&B.")

                # Log Detailed Trials Table using the helper function
                log_detailed_trials_table_to_wandb(study, wandb.run)

            except Exception as e_log:
                 logging.error(f"Rank 0: Error during logging visualizations or trial table to W&B summary run: {e_log}", exc_info=True)
            finally:
                # Ensure W&B run is finished even if logging fails
                if wandb.run is not None:
                    logging.info(f"Rank 0: Finishing W&B summary run: {wandb.run.name}")
                    wandb.finish()
                else:
                    logging.warning("Rank 0: No active W&B run found to finish in the finally block.")

        except Exception as e_init:
            logging.error(f"Rank 0: Failed to initialize W&B summary run: {e_init}", exc_info=True)
            # Ensure wandb is cleaned up if initialization failed partially
            if wandb.run is not None:
                wandb.finish()

    # All workers log their contribution
    logging.info(f"Worker {worker_id} completed optimization")

    # Generate visualizations if enabled (only rank 0 should do this)
    if worker_id == '0' and config.get("optuna", {}).get("visualization", {}).get("enabled", False):
        if study:
            try:
                from wind_forecasting.utils.optuna_visualization import generate_visualizations
                # Import the path resolution helper from db_utils or optuna_db_utils
                from wind_forecasting.utils.db_utils import _resolve_path

                vis_config = config["optuna"]["visualization"]

                # Resolve the output directory using the helper function and full config
                default_vis_path = os.path.join(config.get("logging", {}).get("optuna_dir", "logging/optuna"), "visualizations")
                # Pass vis_config as the dict containing 'output_dir', key 'output_dir', and the full 'config'
                visualization_dir = _resolve_path(vis_config, "output_dir", full_config=config, default=default_vis_path)

                if not visualization_dir:
                     logging.error("Rank 0: Could not determine visualization output directory. Skipping visualization.")
                else:
                    logging.info(f"Rank 0: Resolved visualization output directory: {visualization_dir}")
                    os.makedirs(visualization_dir, exist_ok=True) # Ensure directory exists

                    # Generate plots
                    logging.info(f"Rank 0: Generating Optuna visualizations in {visualization_dir}")
                    summary_path = generate_visualizations(study, visualization_dir, vis_config) # Pass vis_config

                    if summary_path:
                        logging.info(f"Rank 0: Generated Optuna visualizations - summary available at: {summary_path}")
                    else:
                        logging.warning("Rank 0: No visualizations were generated - study may not have enough completed trials or an error occurred.")

            except ImportError:
                 logging.warning("Rank 0: Could not import visualization modules. Skipping visualization generation.")
            except Exception as e:
                logging.error(f"Rank 0: Failed to generate Optuna visualizations: {e}", exc_info=True)
        else:
             logging.warning("Rank 0: Study object not available, cannot generate visualizations.")

    # Log best trial details (only rank 0)
    if worker_id == '0' and study: # Check if study object exists
        if len(study.trials) > 0:
            logging.info("Number of finished trials: {}".format(len(study.trials)))
            logging.info("Best trial:")
            try: # Add try-except for best_trial access
                trial = study.best_trial
                logging.info("  Value: {}".format(trial.value))
                logging.info("  Params: ")
                for key, value in trial.params.items():
                    logging.info("    {}: {}".format(key, value))
            except ValueError:
                 logging.warning("Rank 0: Could not retrieve best trial (likely no trials completed successfully).")
        else:
            logging.warning("No trials were completed")


    if worker_id == '0' and study and len(study.trials) > 0:
        try:
            return study.best_params
        except ValueError:
            logging.warning("Rank 0: Returning empty dict as best_params because best_trial could not be retrieved.")
            return {}
    elif worker_id == '0':
         logging.warning("Rank 0: Returning empty dict as best_params because study has no completed trials.")
         return {}
    else:
         return None # Or {}