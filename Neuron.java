import java.util.HashMap;

//Neuron class for use in a Neural Network
public class Neuron implements Cloneable{

	//makes sure individual Neurons have different id's for reference
	static int idIterator = 0;
	
	//reference id of Neuron
	private int id;
	//value of Neuron
	private float value;
	//output value of Neuron
	private float output;
	//weightmap of outputs, using Neuron id and weight value
	HashMap<Integer, Float> outputWeight;
	
	//Neuron constructor, creates new Neuron and iterates the id
	public Neuron(){
		this.id = idIterator;
		idIterator++;
		outputWeight = new HashMap<Integer, Float>();
	}
	
	//Neuron constructor, creates new Neuron of specified id
	public Neuron(int id){
		this.id = id;
		outputWeight = new HashMap<Integer, Float>();
	}
	
	//Returns id of Neuron
	public int getId(){
		return id;
	}
	//Sets value of Neuron
	public float setValue(float value){
		this.value = value;
		return calcOutput(value);
	}
	//Returns value of Neuron
	public float getValue(){
		return value;
	}
	//Returns output of Neuron
	public float getOutput(){
		return output;
	}
	//Sets the weight of the output of the Neuron to the Neuron of specified id
	public void setOutputWeight(int id, float weight){
		this.outputWeight.put(id, weight);
	}
	//Gets the weight of the output of the Neuron to the Neuron of specified id
	public float getOutputWeight(int id){
		return outputWeight.get(id);
	}
	//Randomizes the weight of the output of the Neuron to a specified set of Neurons
	public void randomOutputWeight(int[] id){
		for(int i = 0; i < id.length; i++){
			outputWeight.put(id[i], (float) (Math.random() * 2) - 1);
		}
	}
	//Randomizes the weight of the output of the Neuron to a specified Neuron
	public void randomOutputWeight(int id){
		outputWeight.put(id, (float) (Math.random() * 2) - 1);
	}
	//Calculates the value of the Neuron from a given set of inputs, and outputs the output value
	float calcValue(Neuron[] inputs){
		float value = 0;
		for(int i = 0; i < inputs.length; i++){
			value += inputs[i].getOutput() * inputs[i].getOutputWeight(id);
		}
		this.value = value;
		calcOutput(value);
		return output;
	}
	//Calculates the output value of the Neuron from a given value
	float calcOutput(float value){
		output = (float) Math.tanh(value);
		return output;
	}
	//Returns a new Neuron with the same output weightmap as the original
	public Neuron clone(){
		try {
			Neuron clone = (Neuron) super.clone();
			clone.outputWeight = new HashMap<Integer, Float>(outputWeight);
			return clone;
		} catch (CloneNotSupportedException e) {
			e.printStackTrace();
		}
		return null;
	}
	
}
