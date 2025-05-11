use std::{cell::RefCell, env, path::PathBuf, rc::Rc};

use awesomlearn::machine_learning::{
    dataloader::MnistDataset,
    nn::{
        backends::wgpu_backend::WgpuBackend,
        dimensions::Dimensions2D,
        model_config::{ActivationType, PoolType},
        nn_model::NNModel,
        nn_model_builder::NNModelBuilder,
        optimizer::AdamOptimizer,
        training_parameters::{LossFunction, TrainingParameters},
    },
};

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() != 2 {
        eprintln!("Usage: {} <path_to_dataset>", args[0]);
        std::process::exit(1);
    }

    let dataset_path = PathBuf::from(&args[1]);

    let dummy_train_dataset = MnistDataset::load_train(dataset_path.clone()).unwrap();
    let (mut dummy_train_dataset, mut dummy_validation_dataset) =
        dummy_train_dataset.random_split(0.8);
    let mut dummy_test_dataset = MnistDataset::load_test(dataset_path).unwrap();
    let batch_size = 128;

    let builder = NNModelBuilder::new(batch_size);
    let config = builder
        .input(1, 28, 28)
        .conv2d(Some(ActivationType::Relu), (3, 3), 32, true)
        .pool2d(PoolType::MaxPool2D, Dimensions2D::new(2, 2))
        .conv2d(Some(ActivationType::Relu), (3, 3), 64, true)
        .pool2d(PoolType::MaxPool2D, Dimensions2D::new(2, 2))
        .flatten()
        .dense(Some(ActivationType::Relu), 1024)
        .dense(Some(ActivationType::Softmax), 10)
        .get_config();

    let backend = WgpuBackend::default();
    let mut model =
        NNModel::from_config(config.clone(), Rc::new(RefCell::new(backend)), batch_size).unwrap();
    let optimizer =
        AdamOptimizer::new(model.backend(), 0.9, 0.999, 10e-8, Some(1e-3), None).unwrap();
    let training_parameters = TrainingParameters {
        epochs: 10,
        loss_function: LossFunction::MSE,
        learning_rate: 0.0001,
    };
    let (_, accuracy_before) = model.predict(&mut dummy_test_dataset).unwrap();
    // let accuracy_before = 0.0;
    let _training_result = model
        .train(
            training_parameters,
            optimizer,
            &mut dummy_train_dataset,
            Some(&mut dummy_validation_dataset),
        )
        .unwrap();
    let (_, accuracy_after) = model.predict(&mut dummy_test_dataset).unwrap();
    println!(
        "Networks accuracy before training: {}, after training: {}",
        accuracy_before, accuracy_after
    );
    model.as_config().unwrap().save("mnist.json").unwrap();
}
