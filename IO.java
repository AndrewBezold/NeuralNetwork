import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;


public class IO {

	public static void output(Genetics genetics, String filename){
		try{
			new File("Genetic/Trial" + Genetics.trial).mkdirs();
			File file = new File("Genetic/Trial" + Genetics.trial + "/" + filename);
			BufferedWriter writer = new BufferedWriter(new FileWriter(file));
			
			String output = genetics.outputPop();
			writer.write(output);
			writer.close();
		}catch(Exception e){
			e.printStackTrace();
		}
	}
	
	public static String output(BasicNN network){
		String encoded = "" + network.inputLayer.length;
		encoded += " " + network.hiddenLayer.length;
		encoded += " " + network.outputLayer.length;
		for(int i = 0; i < network.inputLayer.length; i++){
			for(int j = 0; j < network.hiddenLayer.length; j++){
				encoded += " " + network.inputLayer[i].getOutputWeight(network.hiddenLayer[j].getId());
			}
		}
		for(int i = 0; i < network.hiddenLayer.length; i++){
			for(int j = 0; j < network.outputLayer.length; j++){
				encoded += " " + network.hiddenLayer[i].getOutputWeight(network.outputLayer[j].getId());
			}
		}
		
		return encoded;
	}
	
	public static void output(BasicNN network, String filename){
		try{
			File file = new File(filename);
			BufferedWriter writer = new BufferedWriter(new FileWriter(file));
			
			String encoded = "" + network.inputLayer.length;
			encoded += " " + network.hiddenLayer.length;
			encoded += " " + network.outputLayer.length;
			for(int i = 0; i < network.inputLayer.length; i++){
				for(int j = 0; j < network.hiddenLayer.length; j++){
					encoded += " " + network.inputLayer[i].getOutputWeight(network.hiddenLayer[j].getId());
				}
			}
			for(int i = 0; i < network.hiddenLayer.length; i++){
				for(int j = 0; j < network.outputLayer.length; j++){
					encoded += " " + network.hiddenLayer[i].getOutputWeight(network.outputLayer[j].getId());
				}
			}
			
			writer.write(encoded);
			writer.close();
		}catch(Exception e){
			e.printStackTrace();
		}
	}
	
	public static BasicNN input(String filename){
		try{
			BasicNN network;
			File file = new File(filename);
			BufferedReader reader = new BufferedReader(new FileReader(file));
			String encoded = reader.readLine();
			reader.close();
			String[] encodedArray = encoded.split(" ");
			int inputSize = Integer.parseInt(encodedArray[0]);
			int hiddenSize = Integer.parseInt(encodedArray[1]);
			int outputSize = Integer.parseInt(encodedArray[2]);
			int[] weights = new int[(inputSize + outputSize) * hiddenSize];
			for(int i = 3; i < encodedArray.length; i++){
				weights[i - 3] = Integer.parseInt(encodedArray[i]);
			}
			network = new BasicNN(inputSize, hiddenSize, outputSize);
			network.clear();
			for(int i = 0; i < network.inputLayer.length; i++){
				for(int j = 0; j < network.hiddenLayer.length; j++){
					network.inputLayer[i].setOutputWeight(network.hiddenLayer[j].getId(), weights[i*network.hiddenLayer.length + j]);
				}
			}
			for(int i = 0; i < network.hiddenLayer.length; i++){
				for(int j = 0; j < network.outputLayer.length; j++){
					network.hiddenLayer[i].setOutputWeight(network.outputLayer[j].getId(), weights[network.inputLayer.length * network.hiddenLayer.length + i*network.outputLayer.length + j]);
				}
			}
			return network;
		}catch(Exception e){
			e.printStackTrace();
			return null;
		}
	}
}
