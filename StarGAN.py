#Code referrance: https://github.com/KevinYuimin/StarGAN-Tensorflow

import numpy as np;
import tensorflow as tf;
import time, os;
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt;
import matplotlib.gridspec as gridspec;
from Customized_Layers import *;
from Pattern_Feeder import Pattern_Feeder;
import Hyper_Parameters as hp

class StarGAN:
    def __init__(
        self,        
        attribute_File_Path = None,
        image_Files_Dir = None,
        select_Attribute_List = None,
        image_Size = None,
        initial_Filter_Count = None,
        batch_Size = None,
        gan_Loss_Type = None,
        learning_Rate = None,
        extract_Dir = None,
        ):

        self.tf_Session = tf.Session();

        self.pattern_Feeder = Pattern_Feeder(
            attribute_File_Path = attribute_File_Path or hp.attribute_File_Path,
            image_Files_Dir = image_Files_Dir or hp.image_Files_Dir,
            select_Attribute_List = select_Attribute_List or hp.select_Attribute_List,
            image_Size = image_Size or hp.image_Size,
            batch_Size = batch_Size or hp.batch_Size,
            max_Queue = 100
            )

        self.select_Attribute_List = self.pattern_Feeder.select_Attribute_List;
        self.image_Size = image_Size or hp.image_Size;
        self.initial_Filter_Count = initial_Filter_Count or hp.initial_Filter_Count;
        self.gan_Loss_Type = gan_Loss_Type or hp.gan_Loss_Type;
        self.learning_Rate = learning_Rate or hp.learning_Rate;
        self.extract_Dir = extract_Dir or hp.extract_Dir;

        self.Tensor_Generate();
        self.tf_Saver = tf.train.Saver();

    def Tensor_Generate(self):
        placeholder_Dict = self.pattern_Feeder.placeholder_Dict;

        fake_Image = Generator(
            image_Pattern = placeholder_Dict["Image"], 
            attribute_Pattern = placeholder_Dict["Fake_Attribute"],
            initial_Filter_Count = self.initial_Filter_Count
            )

        reconstructed_Image = Generator(
            image_Pattern = fake_Image, 
            attribute_Pattern = placeholder_Dict["Original_Attribute"],
            initial_Filter_Count = self.initial_Filter_Count,
            reuse = True
            )

        discrimination_Logit_for_Original, attribute_Logit_for_Original = Discriminator(
            image_Pattern = placeholder_Dict["Image"],
            initial_Filter_Count = self.initial_Filter_Count,
            attribute_Count = len(self.select_Attribute_List)
            )

        discrimination_Logit_for_Fake, attribute_Logit_for_Fake = Discriminator(
            image_Pattern = fake_Image,
            initial_Filter_Count = self.initial_Filter_Count,
            attribute_Count = len(self.select_Attribute_List),
            reuse = True
            )


        loss_Dict = {};
                
        #WGAN loss
        lambda_Gradient_Penalty = 10;
        loss_Dict["WGAN", "Discrimination"] = lambda_Gradient_Penalty * WGAN_Gradient_Penalty_Loss(
            original_Image= placeholder_Dict["Image"],
            fake_Image= fake_Image,
            initial_Filter_Count = self.initial_Filter_Count,
            attribute_Count = len(self.select_Attribute_List),
            discriminator_Reuse= True
            ) + tf.reduce_mean(discrimination_Logit_for_Fake) - tf.reduce_mean(discrimination_Logit_for_Original)

        loss_Dict["WGAN", "Generator"] = -tf.reduce_mean(discrimination_Logit_for_Fake)

        #GAN loss
        loss_Dict["GAN", "Discrimination", "Original"] = tf.reduce_mean(
           tf.nn.sigmoid_cross_entropy_with_logits(
               logits = discrimination_Logit_for_Original,
               labels = tf.ones_like(discrimination_Logit_for_Original)
               )
           )
        loss_Dict["GAN", "Discrimination", "Fake"] = tf.reduce_mean(
           tf.nn.sigmoid_cross_entropy_with_logits(
               logits = discrimination_Logit_for_Fake,
               labels = tf.zeros_like(discrimination_Logit_for_Fake)
               )
           )
        
        loss_Dict["GAN", "Discrimination"] = loss_Dict["GAN", "Discrimination", "Original"] + loss_Dict["GAN", "Discrimination", "Fake"]

        loss_Dict["GAN", "Generator"] = tf.reduce_mean(
           tf.nn.sigmoid_cross_entropy_with_logits(
               logits = discrimination_Logit_for_Fake,
               labels = tf.ones_like(discrimination_Logit_for_Fake)
               )
           )

        #LSGAN loss
        loss_Dict["LSGAN", "Discrimination", "Original"] = tf.reduce_mean(
            tf.pow(tf.ones_like(discrimination_Logit_for_Original) - discrimination_Logit_for_Original, 2)
            )
        loss_Dict["LSGAN", "Discrimination", "Fake"] = tf.reduce_mean(
            tf.pow(tf.zeros_like(discrimination_Logit_for_Fake) - discrimination_Logit_for_Fake, 2)
            )

        loss_Dict["LSGAN", "Discrimination"] = loss_Dict["LSGAN", "Discrimination", "Original"] + loss_Dict["LSGAN", "Discrimination", "Fake"]

        loss_Dict["LSGAN", "Generator"] = tf.reduce_mean(
            tf.pow(tf.ones_like(discrimination_Logit_for_Fake) - discrimination_Logit_for_Fake, 2)
            )
        
        #Attribute loss, sum function gives more weight to attribute.
        loss_Dict["Attribute", "Original"] = tf.reduce_mean(tf.reduce_sum(
           tf.nn.sigmoid_cross_entropy_with_logits(
               logits = attribute_Logit_for_Original,
               labels = placeholder_Dict["Original_Attribute"]
               ),
           axis=1
           ))

        loss_Dict["Attribute", "Fake"] = tf.reduce_mean(tf.reduce_sum(
           tf.nn.sigmoid_cross_entropy_with_logits(
               logits = attribute_Logit_for_Fake,
               labels = placeholder_Dict["Fake_Attribute"]
               ),
           axis=1
           ))

        #Reconstruction loss
        lambda_Reconstruction = 10;
        loss_Dict["Reconstruction"] = lambda_Reconstruction * tf.reduce_mean(tf.abs(placeholder_Dict["Image"] - reconstructed_Image));


        discriminator_Loss = loss_Dict[self.gan_Loss_Type, "Discrimination"] + loss_Dict["Attribute", "Original"];
        generator_Loss = loss_Dict[self.gan_Loss_Type, "Generator"] + loss_Dict["Attribute", "Fake"] + loss_Dict["Reconstruction"];
        
        generator_Variable_List = [x for x in tf.trainable_variables() if x.name.startswith('generator')];
        discriminator_Variable_List = [x for x in tf.trainable_variables() if x.name.startswith('discriminator')];

        global_Step = tf.Variable(0, name='global_Step', trainable=False);

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            learning_Rate = tf.train.exponential_decay(
                learning_rate= self.learning_Rate,
                global_step= global_Step,
                decay_steps= len(self.pattern_Feeder.attribute_Dict) // self.pattern_Feeder.batch_Size,
                decay_rate= 0.95
                )

            discriminator_Optimize = tf.train.AdamOptimizer(learning_Rate, beta1=0.5, beta2=0.999).minimize(
                discriminator_Loss,
                var_list= discriminator_Variable_List,
                global_step = global_Step
                )

            generator_Optimize = tf.train.AdamOptimizer(learning_Rate, beta1=0.5, beta2=0.999).minimize(
                generator_Loss, 
                var_list= generator_Variable_List
                )

        self.training_Tensor_List_for_Disciriminator = [global_Step, learning_Rate, tf.reduce_mean(tf.nn.sigmoid(discrimination_Logit_for_Original)), discriminator_Optimize];        
        self.training_Tensor_List_for_Generator = [global_Step, learning_Rate, tf.reduce_mean(tf.nn.sigmoid(discrimination_Logit_for_Fake)), generator_Optimize];        
        self.test_Tensor_List = [global_Step, fake_Image];

        summary_Dir = self.extract_Dir + "/Summary";
        if not os.path.exists(summary_Dir):
            os.makedirs(summary_Dir);
        graph_Writer = tf.summary.FileWriter(summary_Dir, self.tf_Session.graph);
        graph_Writer.close();

        self.tf_Session.run(tf.global_variables_initializer());

    def Restore(self):
        checkpoint_Dir = self.extract_Dir + "/Checkpoint";
        checkpoint_Path = tf.train.latest_checkpoint(checkpoint_Dir);

        if checkpoint_Path is None:
            print("There is no checkpoint");
        else:
            self.tf_Saver.restore(self.tf_Session, checkpoint_Path);
            print("Checkpoint '", checkpoint_Path, "' is loaded.");

    def Train(
        self,
        test_File_Name_List = None,
        test_Attribute_Pattern_List = None,
        generator_Step = 5,
        test_Step = 1000,
        checkpoint_Step = 1000
        ):
        checkpoint_Dir = self.extract_Dir + "/Checkpoint";
        if not os.path.exists(checkpoint_Dir):
            os.makedirs(checkpoint_Dir);
        checkpoint_Path = checkpoint_Dir + '/Checkpoint';

        self.Test(test_File_Name_List, test_Attribute_Pattern_List);
        try:
            while True:
                start_Time = time.time();
                
                feed_Dict = self.pattern_Feeder.Get_Pattern();

                global_Step, learning_Rate, discriminator_Loss, _, = self.tf_Session.run(self.training_Tensor_List_for_Disciriminator, feed_dict = feed_Dict);

                if global_Step % generator_Step == 0:
                    global_Step, learning_Rate, generator_Loss, _, = self.tf_Session.run(self.training_Tensor_List_for_Generator, feed_dict = feed_Dict);
                    print(
                        "Spent_Time: {:.3f}".format(time.time() - start_Time, 3), "\t",
                        "Golbal_Step: {:d}".format(global_Step), "\t",
                        "Learning_Rate: {:.7f}".format(learning_Rate), "\t",
                        "Avg_Original_Decision: {:.5f}".format(discriminator_Loss), "\t",
                        "Avg_Fake_Decision: {:.5f}".format(generator_Loss)
                    )
                else:
                    print(
                        "Spent_Time: {:.3f}".format(time.time() - start_Time, 3), "\t",
                        "Golbal_Step: {:d}".format(global_Step), "\t",
                        "Learning_Rate: {:.7f}".format(learning_Rate), "\t",
                        "Avg_Original_Decision: {:.5f}".format(discriminator_Loss)
                    )
            
                if global_Step % checkpoint_Step == 0:
                    self.tf_Saver.save(self.tf_Session, checkpoint_Path, global_step=global_Step);
                    print("Checkpoint saved");

                if global_Step % test_Step == 0:
                    self.Test(test_File_Name_List, test_Attribute_Pattern_List);

        except KeyboardInterrupt:
            self.tf_Saver.save(self.tf_Session, checkpoint_Path, global_step=global_Step);
            self.Test(test_File_Name_List, test_Attribute_Pattern_List);

    def Test(self, file_Name_List, attribute_Pattern_List, export_File_Name = None):
        test_Dir = self.extract_Dir + "/Result";
        if not os.path.exists(test_Dir):
            os.makedirs(test_Dir);

        original_Image, feed_Dict = self.pattern_Feeder.Get_Test_Pattern(
            file_Name_List = file_Name_List,
            attribute_Pattern_List = attribute_Pattern_List
            )

        global_Step, fake_Image = self.tf_Session.run(
            self.test_Tensor_List,
            feed_dict = feed_Dict
            )

        result_Image_Pattern = np.insert(
            fake_Image,
            np.arange(0, fake_Image.shape[0],len(attribute_Pattern_List)),
            original_Image,
            axis= 0
            )

        if export_File_Name is None:
            export_File_Name = "Step_{}.png".format(global_Step);
        elif all([not export_File_Name.lower().endswith(x) for x in [".png", ".jpg", ".jpeg"]]):
            export_File_Name += ".png";


        self.Result_Plot(result_Image_Pattern, len(file_Name_List), len(attribute_Pattern_List) + 1, test_Dir + "/{}".format(export_File_Name));
        
    def Result_Plot(self, image_Pattern, row_Count, column_Count, file_Name):
        if image_Pattern.shape[0] != row_Count * column_Count:
            print("pattern count should be same to (row * column)")
            return False;

        fig = plt.figure()
        dpi = fig.get_dpi()
        fig.set_size_inches(column_Count * self.image_Size * 2 / float(dpi), row_Count * self.image_Size * 2 / float(dpi))

        grid = gridspec.GridSpec(row_Count, column_Count, hspace=0.0, wspace=0.0)

        for row_Index in range(row_Count):
            for column_Index in range(column_Count):
                plt.subplot(grid[row_Index, column_Index]);
                plt.axis('off');
                plt.gca().set_xticklabels([]);
                plt.gca().set_yticklabels([]);
                plt.gca().set_aspect('equal');
                plt.imshow(self.pattern_Feeder.Pattern_to_Image(image_Pattern[row_Index * column_Count + column_Index]));

        plt.tight_layout()
        plt.savefig(file_Name, bbox_inches='tight');
        plt.close();


if __name__ == "__main__":
    new_StarGAN = StarGAN();
    new_StarGAN.Restore();

    new_StarGAN.Train(
        test_File_Name_List = hp.test_File_Name_List,
        test_Attribute_Pattern_List = hp.test_Attribute_Pattern_List,
        test_Step = hp.test_Step,
        checkpoint_Step = hp.checkpoint_Step
        )
    #new_StarGAN.Test(
    #    file_Name_List=[
    #        "D:/Work&Study/Visiting Scholarship/Photo.jpg"
    #        ],
    #    attribute_Pattern_List=[
    #        np.array([1, 0, 0, 1, 0, 0]),
    #        np.array([0, 1, 0, 1, 0, 0]),
    #        np.array([0, 0, 1, 1, 0, 0]),
    #        np.array([1, 0, 0, 0, 0, 0]),
    #        np.array([1, 0, 0, 0, 1, 0]),
    #        np.array([0, 1, 0, 0, 1, 0]),
    #        ]
    #    )
    file_Name_List=[
        'D:/Work&Study/Visiting Scholarship/Photo.jpg'
        ]

    attribute_Pattern_List=[
        np.array([1, 0, 0, 1, 0, 0]),
        np.array([0, 1, 0, 1, 0, 0]),
        np.array([0, 0, 1, 1, 0, 0]),
        np.array([1, 0, 0, 0, 0, 0]),
        np.array([1, 0, 0, 0, 1, 0]),
        np.array([0, 1, 0, 0, 1, 0])
        ]