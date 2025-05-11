mod awesomaze;

use std::{io, path::PathBuf};

use awesomlearn::machine_learning::reinforcement_learning::q_learning::dqnet::{
    DQNetTrainingConfig, LearningRateScheduler,
};
use clap::{command, Args, Parser, Subcommand, ValueEnum};

#[derive(Parser)]
#[command(about)]
pub struct AwesomazeArgs {
    #[command(subcommand)]
    mode: AwesomazeMode,
}

#[derive(Subcommand)]
pub enum AwesomazeMode {
    Play,
    Watch {
        file: PathBuf,
        #[arg(value_enum, default_value_t = ModelType::Burn )]
        model: ModelType,
        #[arg(value_enum, default_value_t = Difficulty::Medium)]
        difficulty: Difficulty,
    },
    Train {
        #[command(flatten)]
        training_args: DQNetTrainingArgs,
    },
}

#[derive(ValueEnum, Clone, Debug)]
pub enum Difficulty {
    Easy,
    Medium,
    Hard,
}

#[derive(ValueEnum, Clone, Debug)]
pub enum ModelType {
    Burn,
    Awesom,
}

impl Default for ModelType {
    fn default() -> Self {
        Self::Burn
    }
}

impl Default for Difficulty {
    fn default() -> Self {
        Self::Medium
    }
}

#[derive(Args)]
pub struct TrainingParams {
    save_file: PathBuf,
}

#[derive(Parser, Debug)]
pub struct DQNetTrainingArgs {
    // Which implementations to use
    #[arg(value_enum, long, default_value_t = ModelType::Burn )]
    model: ModelType,
    // Difficulty (max size) of the maze
    #[arg(value_enum, long, default_value_t = Difficulty::Medium)]
    difficulty: Difficulty,
    /// Number of parallel environments
    #[arg(long, default_value_t = 32)]
    pub parallel_env: usize,

    /// Maximum number of training steps
    #[arg(long, default_value_t = 1_000_000)]
    pub max_steps: usize,

    /// Early stopping patience
    #[arg(long, default_value_t = 10.0)]
    pub patience: f32,

    /// Batch size for training
    #[arg(long, default_value_t = 32)]
    pub batch_size: usize,

    /// Exploration rate (epsilon)
    #[arg(long, default_value_t = 0.1)]
    pub epsilon: f32,

    /// Target network update rate (tau)
    #[arg(long, default_value_t = 0.005)]
    pub tau: f32,

    /// Size of the replay buffer
    #[arg(long, default_value_t = 100_000)]
    pub replay_buffer_size: usize,

    /// Discount factor (gamma)
    #[arg(long, default_value_t = 0.99)]
    pub gamma: f32,

    #[arg(long)]
    pub load_model: Option<PathBuf>,

    #[arg(long)]
    pub save_model: Option<PathBuf>,

    /// Learning rate
    #[arg(long, default_value_t = 0.0001)]
    lr: f32,

    /// Learning rate decay step
    #[arg(long, default_value_t = 0.98)]
    step_size: f32,

    /// Learning rate decay interval
    #[arg(long, default_value_t = 10_000)]
    update_interval: usize,

    /// Warmup steps
    #[arg(long, default_value_t = 4000)]
    warmup: usize,
}

impl From<DQNetTrainingArgs> for (LearningRateScheduler, DQNetTrainingConfig) {
    fn from(value: DQNetTrainingArgs) -> Self {
        let save_model_name = match value.save_model {
            Some(save_name) => save_name,
            None => {
                let model_type = match value.model {
                    ModelType::Burn => "burn".to_string(),
                    ModelType::Awesom => "awesom".to_string(),
                };
                let difficulty = match value.difficulty {
                    Difficulty::Easy => "16x16".to_string(),
                    Difficulty::Medium => "32x32".to_string(),
                    Difficulty::Hard => "64x64".to_string(),
                };
                format!("{model_type}_{difficulty}").into()
            }
        };
        let training_config = DQNetTrainingConfig {
            parallel_env: value.parallel_env,
            max_steps: value.max_steps,
            patience: value.patience,
            batch_size: value.batch_size,
            epsilon: value.epsilon,
            tau: value.tau,
            replay_buffer_size: value.replay_buffer_size,
            gamma: value.gamma,
            load_model: value.load_model,
            save_model: save_model_name,
        };
        let learning_rate_scheduler = LearningRateScheduler::new(
            value.lr,
            value.step_size,
            value.update_interval,
            value.warmup,
        );
        (learning_rate_scheduler, training_config)
    }
}

fn main() -> io::Result<()> {
    let args = AwesomazeArgs::parse();
    awesomaze::game_loop(args)
}
