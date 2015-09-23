import java.util.Arrays;


public class BasicNN implements Network, Cloneable{

	Neuron[] inputLayer;
	Neuron[] hiddenLayer;
	Neuron[] outputLayer;
	float learningRate = .01f;
	float momentum = .9f;
	boolean confidence;
	
	public BasicNN(int inputSize, int hiddenSize, int outputSize){
		inputLayer = new Neuron[inputSize];
		hiddenLayer = new Neuron[hiddenSize];
		outputLayer = new Neuron[outputSize];
		
		for(int i = 0; i < inputSize; i++){
			inputLayer[i] = new Neuron();
		}
		for(int i = 0; i < hiddenSize; i++){
			hiddenLayer[i] = new Neuron();
		}
		for(int i = 0; i < outputSize; i++){
			outputLayer[i] = new Neuron();
		}
		randomize();
	}
	
	public BasicNN(int inputSize, int hiddenSize, int outputSize, float[][] weights){
		inputLayer = new Neuron[inputSize];
		hiddenLayer = new Neuron[hiddenSize];
		outputLayer = new Neuron[outputSize];
		
		for(int i = 0; i < inputSize; i++){
			inputLayer[i] = new Neuron();
		}
		for(int i = 0; i < hiddenSize; i++){
			hiddenLayer[i] = new Neuron();
		}
		for(int i = 0; i < outputSize; i++){
			outputLayer[i] = new Neuron();
		}
		for(int i = 0; i < hiddenSize; i++){
			setNodeWeight(weights[i]);
		}
	}
	
	public void clear(){
		for(int i = 0; i < inputLayer.length; i++){
			inputLayer[i].outputWeight.clear();
		}
		for(int i = 0; i < hiddenLayer.length; i++){
			hiddenLayer[i].outputWeight.clear();
		}
	}
	
	void randomize(){
		for(int i = 0; i < inputLayer.length; i++){
			for(int j = 0; j < hiddenLayer.length; j++){
				inputLayer[i].setOutputWeight(hiddenLayer[j].getId(), (float) (Math.random() * 2) - 1);
			}
		}
		for(int i = 0; i < hiddenLayer.length; i++){
			for(int j = 0; j < outputLayer.length; j++){
				hiddenLayer[i].setOutputWeight(outputLayer[j].getId(), (float) (Math.random() * 2) - 1);
			}
		}
	}
	
	public float[] run(float[] inputs){
		float[] outputs = new float[outputLayer.length];
		
		for(int i = 0; i < inputLayer.length; i++){
			inputLayer[i].setValue(inputs[i]);
		}
		for(int i = 0; i < hiddenLayer.length; i++){
			hiddenLayer[i].calcValue(inputLayer);
		}
		for(int i = 0; i < outputLayer.length; i++){
			outputs[i] = outputLayer[i].calcValue(hiddenLayer);
		}
		return outputs;
	}
	
	public void setNodeWeight(float[] weights){
		//size of array must equal input plus output
		int hiddenNode = -1;
		for(int i = 0; i < hiddenLayer.length; i++){
			if(hiddenLayer[i].outputWeight.isEmpty()){
				hiddenNode = i;
				break;
			}
		}
		if(hiddenNode == -1){
			return;
		}
		for(int i = 0; i < weights.length; i++){
			if(i < inputLayer.length){
				inputLayer[i].setOutputWeight(hiddenLayer[hiddenNode].getId(), weights[i]);
			}else{
				hiddenLayer[hiddenNode].setOutputWeight(outputLayer[i-inputLayer.length].getId(), weights[i]);
			}
		}
	}
	
	public void addHidden(){
		Neuron[] hiddenLayer = new Neuron[this.hiddenLayer.length + 1];
		for(int i = 0; i < this.hiddenLayer.length; i++){
			hiddenLayer[i] = this.hiddenLayer[i].clone();
		}
		hiddenLayer[hiddenLayer.length - 1] = new Neuron();
		int[] outputArray = new int[outputLayer.length];
		for(int i = 0; i < outputLayer.length; i++){
			outputArray[i] = outputLayer[i].getId();
		}
		hiddenLayer[hiddenLayer.length - 1].randomOutputWeight(outputArray);
		for(int i = 0; i < inputLayer.length; i++){
			inputLayer[i].randomOutputWeight(hiddenLayer[hiddenLayer.length - 1].getId());
		}
		this.hiddenLayer = Arrays.copyOf(hiddenLayer, hiddenLayer.length);
	}
	public void deleteHidden(int id){
		int location = -1;
		for(int i = 0; i < hiddenLayer.length; i++){
			if(id == hiddenLayer[i].getId()){
				location = i;
			}
		}
		if(location != -1){
			Neuron[] hiddenLayer = new Neuron[this.hiddenLayer.length - 1];
			int count = 0;
			for(int i = 0; i < this.hiddenLayer.length; i++){
				if(i != location){
					hiddenLayer[count] = this.hiddenLayer[i].clone();
					count++;
				}
			}
			for(int i = 0; i < inputLayer.length; i++){
				inputLayer[i].outputWeight.remove(id);
			}
			this.hiddenLayer = Arrays.copyOf(hiddenLayer, hiddenLayer.length);
		}
	}
	
	
	
	
	double[] runLayer(Neuron[] inputLayer, Neuron[] layer){
		double[] outputs = new double[layer.length];
		
		for(int i = 0; i < layer.length; i++){
			layer[i].calcValue(inputLayer);
		}
		return outputs;
	}
	
	//trainIn is inputs, trainOut is expected output
	public void backpropagation(float[] trainIn, float[] trainOut){
		float[] actualOut = new float[trainIn.length];
		double[] error;
		double[] hiddenError;
		double[][] deltaWeight = new double[outputLayer.length][hiddenLayer.length];
		double[][] deltaHiddenWeight = new double[hiddenLayer.length][inputLayer.length];
		double[][] oldDeltaWeight = new double[outputLayer.length][hiddenLayer.length];
		double[][] oldDeltaHiddenWeight = new double[hiddenLayer.length][inputLayer.length];
		for(int i = 0; i < oldDeltaWeight.length; i++){
			for(int j = 0; j < oldDeltaWeight[i].length; j++){
				oldDeltaWeight[i][j] = 0;
			}
		}
		for(int i = 0; i < oldDeltaHiddenWeight.length; i++){
			for(int j = 0; j < oldDeltaHiddenWeight[i].length; j++){
				oldDeltaHiddenWeight[i][j] = 0;
			}
		}
		//run training set
		actualOut = run(trainIn);
		//find error
		error = new double[actualOut.length];
		//for each output node
		for(int j = 0; j < error.length; j++){
			//difference times derivative of activation function
			error[j] = (trainOut[j] - actualOut[j]) * (1 - Math.pow(Math.tanh(actualOut[j]), 2));
			//backpropagate error
			//change inputs into output nodes
			for(int k = 0; k < hiddenLayer.length; k++){
				deltaWeight[j][k] = learningRate * error[j] * hiddenLayer[k].getOutput() + oldDeltaWeight[j][k] * momentum;
				hiddenLayer[k].setOutputWeight(outputLayer[j].getId(), (float) (deltaWeight[j][k] + hiddenLayer[k].getOutputWeight(outputLayer[j].getId())));
				oldDeltaWeight[j][k] = deltaWeight[j][k];
			}
		
		}
		//for each hidden node
		hiddenError = new double[hiddenLayer.length];
		for(int j = 0; j < hiddenLayer.length; j++){
			double sumError = 0;
			for(int k = 0; k < error.length; k++){
				sumError += error[k] * hiddenLayer[j].getOutputWeight(outputLayer[k].getId());
			}
			hiddenError[j] = (1 - Math.pow(hiddenLayer[j].getOutput(), 2)) * sumError;
			for(int k = 0; k < inputLayer.length; k++){
				deltaHiddenWeight[j][k] = learningRate * hiddenError[j] * inputLayer[k].getOutput() + oldDeltaHiddenWeight[j][k] * momentum;
				inputLayer[k].setOutputWeight(hiddenLayer[j].getId(), (float) (deltaHiddenWeight[j][k] + inputLayer[k].getOutputWeight(hiddenLayer[j].getId())));
				oldDeltaHiddenWeight[j][k] = deltaHiddenWeight[j][k];
			}
		}
		//calculate global error
	}
	
	public BasicNN clone(){
		try {
			BasicNN clone = (BasicNN) super.clone();
			for(int i = 0; i < inputLayer.length; i++){
				clone.inputLayer[i] = inputLayer[i].clone();
			}
			for(int i = 0; i < hiddenLayer.length; i++){
				clone.hiddenLayer[i] = hiddenLayer[i].clone();
			}
			for(int i = 0; i < outputLayer.length; i++){
				clone.outputLayer[i] = outputLayer[i].clone();
			}
			return clone;
		} catch (CloneNotSupportedException e) {
			e.printStackTrace();
		}
		return null;
	}
	
	public void setConfidence(boolean confidence){
		this.confidence = confidence;
	}
	
	public boolean getConfidence(){
		return confidence;
	}
	
	public int getInputLength(){
		return inputLayer.length;
	}

}
