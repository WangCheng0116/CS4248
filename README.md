# CS4248 Group Project

# Baseline 

To run baseline "ZeroCoT", run the following command:  
```
python zeroCoT.py
```
To evaluate the results, please run:
```
python evaluate-v2.0.py eval_dataset.json predictions.json --na-prob-file na_prob.json
```

To run baseline "ZeroShot", please refer to the following section by setting the number of shots to 0.

# To reproduce our results
Our method consists of two arguments, ```model_name``` and ```num_shots```. To run the experiments on ```Llama-3.2-3B-Instruct```, please specify the ```model_name``` as ```"meta-llama/Llama-3.2-3B-Instruct"```. We provide some sample commands:
```
python main.py --model_name "meta-llama/Llama-3.2-3B-Instruct" --num_shots 3
```

> Note that when ```num_shots``` is 0, the method uses zero shots, becoming one of the baselines we used in our experiments.

After generating the answer, separate folders like ```Llama-3.2-3B-Instruct_3shots_topk``` will be created. To evaluate the results, please use the following command with correct directory:
```
python evaluate-v2.0.py Llama-3.2-3B-Instruct_3shots_topk/eval_dataset.json Llama-3.2-3B-Instruct_3shots_topk/predictions.json --na-prob-file Llama-3.2-3B-Instruct_3shots_topk/na_prob.json
```

# Acknowledgement
Models are directly downloaded from Huggingface. LLaMA-3.2-3B-Instruct is from [3B](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) and LLaMA-3.2-1B-Instruct is from [1B](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct).

For the dataset we use, we refer readers to the [paper](https://aclanthology.org/D16-1264.pdf). For the evaluation script, we use the script provided in this [repo](https://github.com/white127/SQUAD-2.0-bidaf/blob/master/evaluate-v2.0.py).
