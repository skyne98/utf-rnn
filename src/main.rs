use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{loss, ops, Linear, Module, Optimizer, VarBuilder, VarMap};

pub mod test;

const VOTE_DIM: usize = 2;
const RESULTS: usize = 1;
const EPOCHS: usize = 10;
const LAYER1_OUT_SIZE: usize = 4;
const LAYER2_OUT_SIZE: usize = 2;
const LEARNING_RATE: f64 = 0.05;

#[derive(Clone)]
pub struct Dataset {
    pub train_votes: Tensor,
    pub train_results: Tensor,
    pub test_votes: Tensor,
    pub test_results: Tensor,
}

struct MultiLevelPerceptron {
    ln1: Linear,
    ln2: Linear,
    ln3: Linear,
}

impl MultiLevelPerceptron {
    fn new(vs: VarBuilder) -> Result<Self> {
        let ln1 = candle_nn::linear(VOTE_DIM, LAYER1_OUT_SIZE, vs.pp("ln1"))?;
        let ln2 = candle_nn::linear(LAYER1_OUT_SIZE, LAYER2_OUT_SIZE, vs.pp("ln2"))?;
        let ln3 = candle_nn::linear(LAYER2_OUT_SIZE, RESULTS + 1, vs.pp("ln3"))?;
        Ok(Self { ln1, ln2, ln3 })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.ln1.forward(xs)?;
        let xs = xs.relu()?;
        let xs = self.ln2.forward(&xs)?;
        let xs = xs.relu()?;
        self.ln3.forward(&xs).map_err(Into::into)
    }
}

fn train(m: Dataset, dev: &Device) -> anyhow::Result<MultiLevelPerceptron> {
    // Move the training results tensor to the specified device (e.g., GPU or CPU).
    let train_results = m.train_results.to_device(dev)?;
    // Move the training votes tensor to the specified device.
    let train_votes = m.train_votes.to_device(dev)?;

    // Create a variable map to store model parameters.
    let varmap = VarMap::new();
    // Create a variable builder from the variable map, specifying the data type and device.
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, dev);
    // Initialize the multi-level perceptron model with the variable builder.
    let model = MultiLevelPerceptron::new(vs.clone())?;

    // Initialize the SGD optimizer with the model's parameters and the learning rate.
    let mut sgd = candle_nn::SGD::new(varmap.all_vars(), LEARNING_RATE)?;

    // Move the test votes tensor to the specified device.
    let test_votes = m.test_votes.to_device(dev)?;
    // Move the test results tensor to the specified device.
    let test_results = m.test_results.to_device(dev)?;

    // Variable to store the final accuracy of the model.
    let mut final_accuracy: f32 = 0.0;

    // Training loop for the specified number of epochs.
    for epoch in 1..EPOCHS + 1 {
        // Forward pass: compute the logits (raw predictions) for the training data.
        let logits = model.forward(&train_votes)?;
        // Apply log softmax to the logits to get log probabilities.
        let log_sm = ops::log_softmax(&logits, D::Minus1)?;
        // Compute the negative log-likelihood loss between the log probabilities and the true labels.
        let loss = loss::nll(&log_sm, &train_results)?;

        // Perform a backward step to update the model parameters using SGD.
        sgd.backward_step(&loss)?;

        // Forward pass: compute the logits for the test data.
        let test_logits = model.forward(&test_votes)?;
        // Compute the number of correct predictions by comparing the predicted labels with the true labels.
        let sum_ok = test_logits
            .argmax(D::Minus1)?
            .eq(&test_results)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()?;
        // Calculate the test accuracy as a percentage.
        let test_accuracy = sum_ok / test_results.dims1()? as f32;
        final_accuracy = 100. * test_accuracy;

        // Print the epoch number, training loss, and test accuracy.
        println!(
            "Epoch: {epoch:3} Train loss: {:8.5} Test accuracy: {:5.2}%",
            loss.to_scalar::<f32>()?,
            final_accuracy
        );

        // If the test accuracy reaches 100%, stop training early.
        if final_accuracy == 100.0 {
            break;
        }
    }

    // If the final accuracy is less than 100%, return an error indicating the model is not trained well enough.
    if final_accuracy < 100.0 {
        Err(anyhow::Error::msg("The model is not trained well enough."))
    } else {
        // Otherwise, return the trained model.
        Ok(model)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let dev = Device::cuda_if_available(0)?;

    let train_votes_vec: Vec<u32> =
        vec![15, 10, 10, 15, 5, 12, 30, 20, 16, 12, 13, 25, 6, 14, 31, 21];
    let train_votes_tensor = Tensor::from_vec(
        train_votes_vec.clone(),
        (train_votes_vec.len() / VOTE_DIM, VOTE_DIM),
        &dev,
    )?
    .to_dtype(DType::F32)?;

    let train_results_vec: Vec<u32> = vec![1, 0, 0, 1, 1, 0, 0, 1];
    let train_results_tensor =
        Tensor::from_vec(train_results_vec, train_votes_vec.len() / VOTE_DIM, &dev)?;

    let test_votes_vec: Vec<u32> = vec![13, 9, 8, 14, 3, 10];
    let test_votes_tensor = Tensor::from_vec(
        test_votes_vec.clone(),
        (test_votes_vec.len() / VOTE_DIM, VOTE_DIM),
        &dev,
    )?
    .to_dtype(DType::F32)?;

    let test_results_vec: Vec<u32> = vec![1, 0, 0];
    let test_results_tensor =
        Tensor::from_vec(test_results_vec.clone(), test_results_vec.len(), &dev)?;

    let m = Dataset {
        train_votes: train_votes_tensor,
        train_results: train_results_tensor,
        test_votes: test_votes_tensor,
        test_results: test_results_tensor,
    };

    let trained_model: MultiLevelPerceptron;
    loop {
        println!("Trying to train neural network.");
        match train(m.clone(), &dev) {
            Ok(model) => {
                trained_model = model;
                break;
            }
            Err(e) => {
                println!("Error: {}", e);
                continue;
            }
        }
    }

    let real_world_votes: Vec<u32> = vec![13, 22];

    let tensor_test_votes =
        Tensor::from_vec(real_world_votes.clone(), (1, VOTE_DIM), &dev)?.to_dtype(DType::F32)?;

    let final_result = trained_model.forward(&tensor_test_votes)?;

    let result = final_result
        .argmax(D::Minus1)?
        .to_dtype(DType::F32)?
        .get(0)
        .map(|x| x.to_scalar::<f32>())??;
    println!("real_life_votes: {:?}", real_world_votes);
    println!("neural_network_prediction_result: {:?}", result);

    Ok(())
}
