// dataloader.rs
// TODO: add a reference to the burn implementation and license

use std::{
    fs::File,
    io::{self, BufReader, ErrorKind, Read},
    path::{Path, PathBuf},
};

use byteorder::{BigEndian, ReadBytesExt};
use rand::seq::SliceRandom;
use rand::thread_rng;

use super::classes_to_one_hot;

pub trait Dataloader {
    fn get(&mut self, amount: usize) -> (Vec<f32>, Vec<f32>, usize);
    fn size(&self) -> usize;
}

pub struct MnistDataset {
    pub curr_index: usize,
    pub images: Vec<Vec<f32>>,
    pub labels: Vec<Vec<f32>>,
}

impl MnistDataset {
    pub fn random_split(&self, split_amount: f64) -> (Self, Self) {
        let mut rng = thread_rng();
        let mut indices = (0..self.images.len()).collect::<Vec<_>>();
        indices.shuffle(&mut rng);
        let split_index: usize = (split_amount * indices.len() as f64) as usize;
        let split1_indices = &indices[0..split_index];
        let split2_indices = &indices[split_index..];

        let mut split1_images = Vec::with_capacity(split1_indices.len());
        let mut split1_labels = Vec::with_capacity(split1_indices.len());
        for ind in split1_indices {
            split1_images.push(self.images[*ind].clone());
            split1_labels.push(self.labels[*ind].clone());
        }
        let split1 = Self {
            curr_index: 0,
            images: split1_images,
            labels: split1_labels,
        };

        let mut split2_images = Vec::with_capacity(split2_indices.len());
        let mut split2_labels = Vec::with_capacity(split2_indices.len());
        for ind in split2_indices {
            split2_images.push(self.images[*ind].clone());
            split2_labels.push(self.labels[*ind].clone());
        }

        let split2 = Self {
            curr_index: 0,
            images: split2_images,
            labels: split2_labels,
        };
        (split1, split2)
    }

    pub fn load_train(parent_path: PathBuf) -> Result<Self, String> {
        let mnist = get_mnist_dataloader(parent_path);
        if let Ok((dataset, labels)) = mnist.read_train_data() {
            let (data_float, labels_float) = convert_to_float(&dataset, &labels);
            let dataset = Self {
                curr_index: 0,
                images: data_float,
                labels: labels_float,
            };
            Ok(dataset)
        } else {
            Err("Couldn't read the mnist train dataset.".to_string())
        }
    }

    pub fn load_test(parent_path: PathBuf) -> Result<Self, String> {
        let mnist = get_mnist_dataloader(parent_path);
        if let Ok((dataset, labels)) = mnist.read_test_data() {
            let (data_float, labels_float) = convert_to_float(&dataset, &labels);
            let dataset = Self {
                curr_index: 0,
                images: data_float,
                labels: labels_float,
            };
            Ok(dataset)
        } else {
            Err("Couldn't read the mnist test dataset.".to_string())
        }
    }
}

impl Dataloader for MnistDataset {
    fn get(&mut self, amount: usize) -> (Vec<f32>, Vec<f32>, usize) {
        let end_index = (self.curr_index + amount).min(self.images.len());
        let batch_features = self.images[self.curr_index..end_index].to_vec();
        let batch_size = end_index - self.curr_index;
        let batch_classes = self.labels[self.curr_index..end_index].to_vec();

        let batch_features = batch_features.into_iter().flatten().collect();
        let batch_classes = batch_classes.into_iter().flatten().collect();
        self.curr_index = if end_index < self.images.len() {
            end_index
        } else {
            0
        };
        (batch_features, batch_classes, batch_size)
    }

    fn size(&self) -> usize {
        self.labels.len()
    }
}

pub struct MnistDataloader {
    training_images_path: PathBuf,
    training_labels_path: PathBuf,
    test_images_path: PathBuf,
    test_labels_path: PathBuf,
}

pub fn get_mnist_dataloader(parent_path: PathBuf) -> MnistDataloader {
    let training_images_filepath = parent_path.join("train/train-images.idx3-ubyte");
    let training_labels_filepath = parent_path.join("train/train-labels.idx1-ubyte");
    let test_images_filepath = parent_path.join("test/t10k-images.idx3-ubyte");
    let test_labels_filepath = parent_path.join("test/t10k-labels.idx1-ubyte");
    MnistDataloader::new(
        training_images_filepath,
        training_labels_filepath,
        test_images_filepath,
        test_labels_filepath,
    )
}

pub fn convert_to_float(data: &[Vec<u8>], labels: &[u8]) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut data_float = Vec::with_capacity(data.len());
    for x in data {
        let mut image = Vec::with_capacity(x.len());
        for pixel in x {
            let pixel_as_float = *pixel as f32 / 255.0;
            image.push(pixel_as_float);
        }
        data_float.push(image);
    }
    let mut labels_float = Vec::with_capacity(labels.len());
    for y in labels {
        let label_as_float = *y as f32;
        labels_float.push(label_as_float);
    }
    let labels_one_hot = classes_to_one_hot(&labels_float);
    (data_float, labels_one_hot)
}

impl MnistDataloader {
    pub fn new(
        training_images_path: PathBuf,
        training_labels_path: PathBuf,
        test_images_path: PathBuf,
        test_labels_path: PathBuf,
    ) -> Self {
        MnistDataloader {
            training_images_path,
            training_labels_path,
            test_images_path,
            test_labels_path,
        }
    }

    fn read_labels<P: AsRef<Path>>(path: P) -> io::Result<Vec<u8>> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let magic = reader.read_u32::<BigEndian>()?;
        if magic != 2049 {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                format!("Magic number mismatch, expected 2049, got {}", magic),
            ));
        }
        let _size = reader.read_u32::<BigEndian>()?;
        let mut labels = Vec::new();
        reader.read_to_end(&mut labels)?;
        Ok(labels)
    }

    fn read_images<P: AsRef<Path>>(path: P) -> io::Result<Vec<Vec<u8>>> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let magic = reader.read_u32::<BigEndian>()?;
        if magic != 2051 {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                format!("Magic number mismatch, expected 2051, got {}", magic),
            ));
        }
        let size = reader.read_u32::<BigEndian>()?;
        let rows = reader.read_u32::<BigEndian>()?;
        let cols = reader.read_u32::<BigEndian>()?;
        let mut image_data = Vec::new();
        reader.read_to_end(&mut image_data)?;

        // Split and reshape the image data
        let images: Vec<Vec<u8>> = image_data
            .chunks_exact((rows * cols) as usize)
            .map(|chunk| chunk.to_vec())
            .collect();

        if images.len() != size as usize {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                "File size and number of images do not match.",
            ));
        }

        Ok(images)
    }

    fn read_images_labels(
        images_filepath: &Path,
        labels_filepath: &Path,
    ) -> io::Result<(Vec<Vec<u8>>, Vec<u8>)> {
        let labels = Self::read_labels(labels_filepath)?;
        let images = Self::read_images(images_filepath)?;
        Ok((images, labels))
    }

    pub fn read_train_data(&self) -> io::Result<(Vec<Vec<u8>>, Vec<u8>)> {
        Self::read_images_labels(&self.training_images_path, &self.training_labels_path)
    }

    pub fn read_test_data(&self) -> io::Result<(Vec<Vec<u8>>, Vec<u8>)> {
        Self::read_images_labels(&self.test_images_path, &self.test_labels_path)
    }
}
