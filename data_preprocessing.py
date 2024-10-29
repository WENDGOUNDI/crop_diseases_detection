# Libraries importation
import splitfolders

# Split with a ratio, train=80% val=10% test=10%
# To split dataset into training validation and testing
# For running this code, we assume you have a single folder containing all classes that will be later splitted into train val and test.
input_folder = "./data" # path to your single folder with all classes
output_folder = "./cassava_split_data"  # path to where you woudl like to save the splitted data

splitfolders.ratio(input_folder, output=output_folder, seed=1337, ratio=(.8, .1, .1)) 

print("[NFO]: Data Splitting Completed")