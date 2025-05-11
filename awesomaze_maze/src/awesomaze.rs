mod agent_engine;
mod engine;
mod game_engine;
mod game_model;
mod info;
mod input;
mod model;
mod render_engine;
mod watch_engine;

use agent_engine::TrainingEngine;
use awesomlearn::machine_learning::nn::{
    backends::wgpu_backend::WgpuBackend, dimensions::Dimensions3D,
};
use burn::backend::{wgpu::WgpuDevice, Wgpu};
use input::handle_events;
use model::{awesom_model, burn_model};
use watch_engine::WatchEngine;

use crate::{AwesomazeArgs, AwesomazeMode, DQNetTrainingArgs, Difficulty, ModelType};

use self::{engine::Engine, game_engine::GameEngine, render_engine::RenderEngine};
use std::{cell::RefCell, io, path::PathBuf, rc::Rc};

pub fn game_loop(game_args: AwesomazeArgs) -> io::Result<()> {
    let mut render_engine = RenderEngine::init_render_engine()?;
    let mut should_quit = false;

    match game_args.mode {
        AwesomazeMode::Play => {
            let mut game_engine = GameEngine::new();
            while !should_quit {
                // rendering
                render_engine.render(|frame| game_engine.render_frame(frame))?;
                // tick
                let user_input = handle_events()?;
                should_quit = game_engine.tick(user_input)?;
            }
        }
        AwesomazeMode::Train { training_args } => {
            match training_args.difficulty {
                Difficulty::Easy => {
                    learning_model_loop::<16, 16>(training_args, &mut render_engine)?
                }
                Difficulty::Medium => {
                    learning_model_loop::<32, 32>(training_args, &mut render_engine)?
                }
                Difficulty::Hard => {
                    learning_model_loop::<64, 64>(training_args, &mut render_engine)?
                }
            };
        }
        AwesomazeMode::Watch {
            model,
            file,
            difficulty,
        } => match difficulty {
            Difficulty::Easy => watch_model_loop::<16, 16>(model, file, &mut render_engine)?,
            Difficulty::Medium => watch_model_loop::<32, 32>(model, file, &mut render_engine)?,
            Difficulty::Hard => watch_model_loop::<64, 64>(model, file, &mut render_engine)?,
        },
    };
    render_engine.deinit_render_engine()?;
    Ok(())
}

pub fn learning_model_loop<const W: usize, const H: usize>(
    training_args: DQNetTrainingArgs,
    render_engine: &mut RenderEngine,
) -> io::Result<()> {
    let model_type = training_args.model.clone();
    let mut should_quit = false;
    let (learningrate_scheduler, training_config) = training_args.into();
    let mut rl_engine =
        TrainingEngine::<W, H>::new(training_config, learningrate_scheduler, model_type);
    while !should_quit {
        // rendering
        render_engine.render(|frame| rl_engine.render_frame(frame))?;
        // tick
        let user_input = handle_events()?;
        should_quit = rl_engine.tick(user_input)?;
    }
    Ok(())
}

pub fn watch_model_loop<const W: usize, const H: usize>(
    model_type: ModelType,
    model_file: PathBuf,
    render_engine: &mut RenderEngine,
) -> io::Result<()> {
    let mut should_quit = false;

    let input_dimensions = Dimensions3D::new(1, H, W);

    match model_type {
        ModelType::Burn => {
            let device = WgpuDevice::default();
            let inference_model = burn_model::InferenceBurnModel::<Wgpu>::load_model(
                model_file,
                input_dimensions,
                device,
            );
            let mut watch_engine = WatchEngine::<W, H, _>::new(inference_model);
            while !should_quit {
                // rendering
                render_engine.render(|frame| watch_engine.render_frame(frame))?;
                // tick
                let user_input = handle_events()?;
                should_quit = watch_engine.tick(user_input)?;
            }
        }
        ModelType::Awesom => {
            let backend = Rc::new(RefCell::new(WgpuBackend::default()));
            let inference_model = awesom_model::AwesomModel::from_file(backend, model_file);
            let mut watch_engine = WatchEngine::<W, H, _>::new(inference_model);
            while !should_quit {
                // rendering
                render_engine.render(|frame| watch_engine.render_frame(frame))?;
                // tick
                let user_input = handle_events()?;
                should_quit = watch_engine.tick(user_input)?;
            }
        }
    };
    Ok(())
}
