import tensorflow as tf;
import numpy as np;
import random, time;
from imageio import imread;
from skimage.transform import resize;
import _thread as thread;
from collections import deque;
import os;

class Pattern_Feeder:
    def __init__(
        self,
        attribute_File_Path = "D:/Simulation_Raw_Data/CelebA/list_attr_celeba.txt",
        image_Files_Dir = "D:/Simulation_Raw_Data/CelebA/img",
        excluded_Pattern_Ratio = 0.1,
        select_Attribute_List = None,
        image_Size = 128,
        batch_Size = 32,
        max_Queue = 100
        ):
        self.attribute_File_Path = attribute_File_Path;
        self.image_Files_Dir = image_Files_Dir;
        self.excluded_Pattern_Ratio = excluded_Pattern_Ratio;
        self.select_Attribute_List = select_Attribute_List;
        self.image_Size = image_Size;
        self.batch_Size = batch_Size;
        self.max_Queue = max_Queue;

        self.Load_Attribute();
        self.Placeholder_Generate();        

        self.pattern_Queue = deque();

        thread.start_new_thread(self.Pattern_Generate, ());

    def Placeholder_Generate(self):
        self.placeholder_Dict = {};
        with tf.variable_scope('placeHolders') as scope:
            self.placeholder_Dict["Image"] = tf.placeholder(tf.float32, shape=(None, self.image_Size, self.image_Size, 3), name="image_Placeholder");    #Shape: [batch_Size, image_Width, image_Height, channels(RGB)];
            self.placeholder_Dict["Original_Attribute"] = tf.placeholder(tf.float32, shape=(None, len(self.select_Attribute_List)), name="original_Attribute_Placeholder");    #[batch_Size, attribute_Size];
            self.placeholder_Dict["Fake_Attribute"] = tf.placeholder(tf.float32, shape=(None, len(self.select_Attribute_List)), name="fake_Attribute_Placeholder");    #[batch_Size, attribute_Size];

    def Load_Attribute(self):
        with open(self.attribute_File_Path, "r") as f:
            readLines = f.readlines()[1:];
        file_Attribute_List = readLines[0].strip().split(' ');
        if self.select_Attribute_List is None:
            self.select_Attribute_List = file_Attribute_List;

        selected_Attribute_Index_List = [file_Attribute_List.index(attribute) for attribute in self.select_Attribute_List];
        self.attribute_Dict = {};

        last_Index = int(len(readLines[1:]) * (1 - self.excluded_Pattern_Ratio))
        print("The number of pattern will be trained: {}".format(last_Index));
        for readLine in readLines[1:][:last_Index]:
            raw_Data = readLine.strip().replace('  ', ' ').split(' ');
            self.attribute_Dict[raw_Data[0].strip()] = np.clip(
                np.array(
                    [int(raw_Data[1:][selected_Attribute_Index]) for selected_Attribute_Index in selected_Attribute_Index_List],
                    dtype=np.float32
                    ),
                a_min= 0,
                a_max= 1
                )
        
    def Pattern_Generate(self):
        while True:
            file_Name_List = list(self.attribute_Dict.keys());            
            random.shuffle(file_Name_List);

            index = 0;
            while index < len(self.attribute_Dict):
                if len(self.pattern_Queue) >= self.max_Queue:
                    time.sleep(0.1)
                    continue;

                image_Pattern_List = [resize(imread(os.path.join(self.image_Files_Dir,file_Name).replace("\\", "/")), (self.image_Size, self.image_Size), mode='constant') * 2 - 1 for file_Name in file_Name_List[index:index+self.batch_Size]];
                image_Pattern_List = [np.flip(image, axis=1) if random.random() > 0.5 else image for image in image_Pattern_List];
                attribute_Pattern_List = [self.attribute_Dict[file_Name] for file_Name in file_Name_List[index:index+self.batch_Size]];                
                fake_Attribute_Pattern_List = np.copy(attribute_Pattern_List);
                random.shuffle(fake_Attribute_Pattern_List)

                image_Pattern = np.stack(image_Pattern_List, axis=0).astype(np.float32);
                original_Attribute_Pattern = np.stack(attribute_Pattern_List, axis=0).astype(np.float32);
                fake_Attribute_Pattern = np.stack(fake_Attribute_Pattern_List, axis=0).astype(np.float32);
                
                new_Feed_Dict = {
                    self.placeholder_Dict["Image"]: image_Pattern,
                    self.placeholder_Dict["Original_Attribute"]: original_Attribute_Pattern,
                    self.placeholder_Dict["Fake_Attribute"]: fake_Attribute_Pattern
                    }

                self.pattern_Queue.append(new_Feed_Dict);                
                index += self.batch_Size;

    def Get_Pattern(self):
        while len(self.pattern_Queue) == 0:
            time.sleep(0.1);
        return self.pattern_Queue.popleft();

    def Get_Test_Pattern(self, file_Name_List, attribute_Pattern_List):
        if not all([len(self.select_Attribute_List) == attribute_Pattern.shape[0] for attribute_Pattern in attribute_Pattern_List]):
            return False;

        loaded_Image_Pattern_List = [resize(imread(file_Name), (self.image_Size, self.image_Size), mode='constant') * 2 - 1 for file_Name in file_Name_List];
        
        image_Pattern_List = [x for x in loaded_Image_Pattern_List for index in range(len(attribute_Pattern_List))]
        attribute_Pattern_List = attribute_Pattern_List * len(file_Name_List);
        
        loaded_Image_Pattern = np.stack(loaded_Image_Pattern_List, axis=0).astype(np.float32);
        image_Pattern = np.stack(image_Pattern_List, axis=0).astype(np.float32);
        attribute_Pattern = np.stack(attribute_Pattern_List, axis=0).astype(np.float32);

        new_Feed_Dict = {
            self.placeholder_Dict["Image"]: image_Pattern,
            self.placeholder_Dict["Fake_Attribute"]: attribute_Pattern
            }

        return loaded_Image_Pattern, new_Feed_Dict;

    def Pattern_to_Image(self, pattern):
        return (pattern + 1) / 2;

if __name__ == "__main__":
    x = Pattern_Feeder(select_Attribute_List=["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young", "Pale_Skin"]);
    while True:
        time.sleep(0.1);
        t = x.Get_Pattern()[x.placeholder_Dict["Image"]];
        print(t.shape);

    #t = x.Get_Test_Pattern(
    #    file_Name_List = [
    #        "D:/Simulation_Raw_Data/CelebA/Img/000047.jpg",
    #        "D:/Simulation_Raw_Data/CelebA/Img/002047.jpg",
    #        "D:/Simulation_Raw_Data/CelebA/Img/000547.jpg",
    #        "D:/Simulation_Raw_Data/CelebA/Img/010043.jpg",
    #        "D:/Simulation_Raw_Data/CelebA/Img/000127.jpg",
    #        ],
    #    attribute_Pattern_List = [
    #        np.array([0,1,1,1,0,0]),
    #        np.array([0,1,1,0,1,1]),
    #        np.array([1,0,0,0,0,1])
    #        ]
    #    )
    #print(t[x.placeholder_Dict["Fake_Attribute"]].shape)
    #print(t[x.placeholder_Dict["Fake_Attribute"]][1])
    #print(t[x.placeholder_Dict["Fake_Attribute"]][2])
    #print(t[x.placeholder_Dict["Fake_Attribute"]][3])