use candle_core::{DType, Device, Tensor, D};
use candle_nn::{loss, ops, Linear, Module, Optimizer, VarBuilder, VarMap};

use crate::{train, Dataset, MultiLevelPerceptron, VOTE_DIM};

#[tokio::test]
async fn simplified() -> anyhow::Result<()> {
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
