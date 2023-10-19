import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import metrics


class Voting():
    
    def __init__(self, predictions, dict_mints):
        """ predictions: list of tupels with form (prediction_data_path, modelname+obv/rev/comb)
        dict_mints: dict_mints.txt as dict
        """
        self.predictions = predictions
        self.dict_mints = dict_mints
        self.predictions_array = []
      
        
    def load_prediction_arrays(self):
        for prediction in self.predictions: 
            pred = np.load(prediction[0])
            self.predictions_array.append((prediction[1],pred))
    
    def top_x_predictions(self, x, prediction, modelname):
        top_x_indices = np.argsort(prediction)[-x:]
        savings = []
        
        for index in top_x_indices:
            class_name = self.dict_mints[index]
            probability = prediction[index]
            savings.append((class_name, probability))
      
        return savings
    
    
    def majority_voter_unweighted(self, model_predictions):
        all_predicted_classes = {} 
        
        for predictions in model_predictions:
           for predicted_classes in predictions[1]:
               if predicted_classes[1] > 0: #only give a vot if prob > 0
                   if predicted_classes[0] in all_predicted_classes.keys():
                       x = all_predicted_classes.get(predicted_classes[0])
                       new = x + 1
                       all_predicted_classes.update({predicted_classes[0]: new})
                   else:
                       all_predicted_classes.update({predicted_classes[0]: 1})
                   
        max_votes = 0
        for key, value in all_predicted_classes.items():
            if value > max_votes:
                max_votes = value
            else:
                continue
       
        most_voted = [] 
        for key, value in all_predicted_classes.items():
            if value == max_votes:
                most_voted.append((key, value))
        if len(most_voted) > 1:
            most_voted = self.majority_voter_weighted(model_predictions)

        return most_voted
    
    def majority_voter_weighted(self, model_predictions):
        """

        model_predictions: list of tupels: (model, top_x_predictions)

        """
        all_predicted_classes = {} 
        
        for predictions in model_predictions:
           for predicted_classes in predictions[1]:
               if predicted_classes[0] in all_predicted_classes.keys():
                   x = all_predicted_classes.get(predicted_classes[0])
                   new = x + predictions[1].index(predicted_classes) +1
                   all_predicted_classes.update({predicted_classes[0]: new})
               else:
                   value = predictions[1].index(predicted_classes) +1
                   all_predicted_classes.update({predicted_classes[0]: value})
                   
        max_votes = 0
        for key, value in all_predicted_classes.items():
            if value > max_votes:
                max_votes = value
            else:
                continue
       
        most_voted = [] 
        for key, value in all_predicted_classes.items():
            if value == max_votes:
                most_voted.append((key, value)) 
       
        if len(most_voted) > 1:
            most_voted = self.prob_majotity_voting(model_predictions)
        
        return most_voted
        
    def prob_majotity_voting(self, model_predictions):
        """

        model_prediction: list of tupels: (model, top_x_predictions)

        """
            
        all_predicted_classes = {}  # this dictionary saves the summed probability of the predicted classes
            
        for predictions in model_predictions:
            for predicted_classes in predictions[1]:
                if predicted_classes[0] in all_predicted_classes.keys():
                    x = all_predicted_classes.get(predicted_classes[0])
                    new = x + predicted_classes[1]
                    all_predicted_classes.update({predicted_classes[0]: new})
                else:
                    all_predicted_classes.update({predicted_classes[0]: predicted_classes[1]})
                    
        max_prob = 0
        class_name = ''
        for key, value in all_predicted_classes.items():
            if value > max_prob:
                max_prob = value
                class_name = key
            else:
                continue
                
        return [class_name, max_prob]
   
    
    
    def run_voting(self, plot = True):
        
        prediction_count = {"images": 0, "wmprediction": 0, "uwmprediction": 0, "pmprediction": 0}
        mints_images = {}
        right_mints_wmprediction = {}
        right_mints_uwmprediction = {}
        right_mints_pmprediction = {}
        for value in self.dict_mints.values():
            mints_images.update({value: 0})
            right_mints_wmprediction.update({value: 0})
            right_mints_uwmprediction.update({value: 0})
            right_mints_pmprediction.update({value: 0})
        
        # das ist jetzt für die netzte die wir haben, könnte man noch anpassen, dass das nach den eingegebenen netzen gemacht wird
        right_mints_resnetobv = mints_images.copy()
        right_mints_resnetrev = mints_images.copy()
        right_mints_resnetboth = mints_images.copy()
        right_mints_vggobv = mints_images.copy()
        right_mints_vggrev = mints_images.copy()
        right_mints_vggboth = mints_images.copy()
        right_mints_resnet101obv = mints_images.copy()
        right_mints_resnet101rev = mints_images.copy()
        right_mints_resnet101both = mints_images.copy()
        
        
        wm_predictions = []
        uwm_predictions = []
        pm_predictions = []
        true_labels = []
        
        
        self.load_prediction_arrays()
        for entry in self.predictions_array: 
            prediction_count.update({entry[0]: 0})
        count = 0 
        len_count = []
        for i in range (0, len(self.predictions_array[0][1])):
            prediction_count.update({"images": prediction_count["images"]+1})
            all_predictions = []
            for prediction in self.predictions_array:
                true_label = self.dict_mints[list(prediction[1][i][1]).index(1)]
                #in der nächsten Zeile top_x_predictions erste Zahl ändern 
                all_predictions.append((prediction[0],self.top_x_predictions(5, prediction[1][i][0], prediction[0])))
            mints_images.update({true_label: mints_images[true_label]+1})
            wmprediction = self.majority_voter_weighted(all_predictions)
            uwmprediction = self.majority_voter_unweighted(all_predictions)
            pmprediction = self.prob_majotity_voting(all_predictions)
            
            wm_predictions.append(wmprediction[0][0])
            uwm_predictions.append(uwmprediction[0][0])
            pm_predictions.append(pmprediction[0])
            true_labels.append(true_label)
          
            
            
            for element in all_predictions:
                if element[1][-1][0] == true_label:
                    prediction_count.update({element[0]: prediction_count[element[0]]+1})
                    
                # das ist jetzt für die netze die wir haben, könnte man noch anpassen, dass das nach den eingegebenen netzen gemacht wird
                if "vgg16obv" in element[0] and element[1][-1][0] == true_label:
                    right_mints_vggobv.update( {true_label: right_mints_vggobv[true_label] +1 })
                if "vgg16both" in element[0] and element[1][-1][0] == true_label:
                    right_mints_vggboth.update( {true_label: right_mints_vggboth[true_label] +1 })
                if "vgg16rev" in element[0] and element[1][-1][0] == true_label:
                     right_mints_vggrev.update( {true_label: right_mints_vggrev[true_label] +1 })
                if "resnet50v2obv" in element[0] and element[1][-1][0] == true_label:
                    right_mints_resnetobv.update( {true_label: right_mints_resnetobv[true_label] +1 })
                if "resnet50v2rev" in element[0] and element[1][-1][0] == true_label:
                    right_mints_resnetrev.update( {true_label: right_mints_resnetrev[true_label] +1 })
                if "resnet50v2both" in element[0] and element[1][-1][0] == true_label:
                    right_mints_resnetboth.update( {true_label: right_mints_resnetboth[true_label] +1 })
                if "resnet101v2rev" in element[0] and element[1][-1][0] == true_label:
                    right_mints_resnet101rev.update( {true_label: right_mints_resnet101rev[true_label] +1 })
                if "resnet101v2obv" in element[0] and element[1][-1][0] == true_label:
                    right_mints_resnet101obv.update( {true_label: right_mints_resnet101obv[true_label] +1 })
                if "resnet101v2both" in element[0] and element[1][-1][0] == true_label:
                    right_mints_resnet101both.update( {true_label: right_mints_resnet101both[true_label] +1 })
                 
                 
            
            if true_label in wmprediction or  true_label in wmprediction[0]:
                prediction_count.update({"wmprediction": prediction_count["wmprediction"]+1})
                right_mints_wmprediction.update({true_label: right_mints_wmprediction[true_label]+1})
            # if len(wmprediction) > 1:
            #     count = count+1
            if true_label in uwmprediction or true_label in uwmprediction[0]:
                prediction_count.update({"uwmprediction": prediction_count["uwmprediction"]+1})
                right_mints_uwmprediction.update({true_label: right_mints_uwmprediction[true_label]+1})
            # if len(uwmprediction) > 1:
            #     count = count+1
            #     len_count.append(len(uwmprediction))
            if true_label in pmprediction:
                prediction_count.update({"pmprediction": prediction_count["pmprediction"]+1})
                right_mints_pmprediction.update({true_label: right_mints_pmprediction[true_label]+1})
            
            
        # print(count, len_count)
        if plot == True:
     
            x = self.plot_results(mints_images, right_mints_wmprediction, right_mints_uwmprediction, right_mints_pmprediction, right_mints_resnetobv, right_mints_resnetrev, right_mints_resnetboth, right_mints_vggobv, right_mints_vggrev, right_mints_vggboth, right_mints_resnet101obv, right_mints_resnet101rev, right_mints_resnet101both)
            
            plt.rcParams['axes.grid'] = False
            confusion_matrix_pm = sklearn.metrics.confusion_matrix(true_labels, pm_predictions, labels = list(mints_images.keys()), normalize='true')
            cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix_pm, display_labels = list(mints_images.keys())) 
            fig, ax = plt.subplots(figsize=(30,30))
            cm_display.plot(ax=ax, xticks_rotation=90,  include_values=False)
            plt.savefig("confusion_matrix_testset_pm.jpg")
 
            confusion_matrix_wm = sklearn.metrics.confusion_matrix(true_labels, wm_predictions, labels = list(mints_images.keys()), normalize='true')
            cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix_wm, display_labels = list(mints_images.keys())) 
            fig, ax = plt.subplots(figsize=(30,30))
            cm_display.plot(ax=ax, xticks_rotation=90, include_values=False)
            plt.savefig("confusion_matrix_testset_wm.jpg")
            
            confusion_matrix_uwm = sklearn.metrics.confusion_matrix(true_labels, uwm_predictions, labels = list(mints_images.keys()), normalize='true')
            cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix_uwm, display_labels = list(mints_images.keys())) 
            fig, ax = plt.subplots(figsize=(30,30))
            cm_display.plot(ax=ax, xticks_rotation=90, include_values=False)
            plt.savefig("confusion_matrix_testset_uwm.jpg")
     
            plt.show() 
            
        print(prediction_count)
            
        return wm_predictions, uwm_predictions, pm_predictions


          
    def plot_results(self, mints_images, right_mints_wmprediction, right_mints_uwmprediction, right_mints_pmprediction, right_mints_resnetobv, right_mints_resnetrev, right_mints_resnetboth, right_mints_vggobv, right_mints_vggrev, right_mints_vggboth, right_mints_resnet101obv, right_mints_resnet101rev, right_mints_resnet101both):
        """ Function only for ploting and saving voting results as bar chart"""
        right_mints_RandomForrest = {'Abdera': 15, 'Abydos': 33, 'Adramyttion': 62, 'Aigospotamoi': 0, 'Ainos': 23, 'Alexandria Troas': 14, 'Alopekonnesos': 0, 'Anchialos': 32, 'Antandros': 15, 'Antiocheia Troas': 7, 'Apollonia Pontica': 0, 'Apollonia ad Rhyndacum': 39, 'Assos': 31, 'Atarneus': 7, 'Attaea': 26, 'Augusta Traiana': 46, 'Birytis': 7, 'Bisanthe': 12, 'Bizye': 37, 'Byzantion': 96, 'Chersonesus Thracica': 7, 'Dardanos': 27, 'Deultum': 1, 'Dikaia': 3, 'Dionysopolis': 8, 'Elaious': 8, 'Gambrion': 18, 'Gargara': 18, 'Gentinos': 1, 'Gergis': 7, 'Germe': 21, 'Hadrianeia': 14, 'Hadrianoi': 4, 'Hadrianopolis': 62, 'Hadrianotherai': 13, 'Hamaxitos': 0, 'Hephaistia': 34, 'Ilion': 48, 'Imbros': 19, 'Iolla': 8, 'Istros': 33, 'Kabyle': 2, 'Kallatis': 19, 'Kardia': 9, 'Kebren': 16, 'Kisthene': 1, 'Koila': 8, 'Kolonai': 1, 'Kypsela': 2, 'Kyzikos': 33, 'Lamponeia': 1, 'Lampsakos': 65, 'Lysimacheia': 14, 'Madytos': 1, 'Markianopolis': 6, 'Maroneia': 17, 'Mesembria': 13, 'Miletoupolis': 32, 'Myrina': 18, 'Neandreia': 7, 'Nikopolis ad Istrum': 16, 'Nikopolis ad Mestum': 0, 'Odessos': 25, 'Ophryneion': 16, 'Orthagoria': 1, 'Parion': 21, 'Pautalia': 1, 'Pergamon': 74, 'Perinthos': 98, 'Perperene': 24, 'Philippopolis': 40, 'Pionia': 19, 'Pitane': 32, 'Plakia': 3, 'Plotinopolis': 2, 'Poimanenon': 11, 'Priapos': 19, 'Prokonnesos': 14, 'Samothrake': 7, 'Selymbria': 14, 'Serdika': 14, 'Sestos': 16, 'Sigeion': 6, 'Skamandria': 13, 'Skepsis': 49, 'Tenedos': 2, 'Thasos': 19, 'Tomis': 4, 'Topeiros': 6, 'Traianopolis': 14, 'Unknown': 44, 'Zeleia': 2}
        right_mints_LogisticRegression = {'Abdera': 21, 'Abydos': 34, 'Adramyttion': 51, 'Aigospotamoi': 0, 'Ainos': 25, 'Alexandria Troas': 13, 'Alopekonnesos': 0, 'Anchialos': 30, 'Antandros': 15, 'Antiocheia Troas': 5, 'Apollonia Pontica': 0, 'Apollonia ad Rhyndacum': 37, 'Assos': 32, 'Atarneus': 8, 'Attaea': 25, 'Augusta Traiana': 47, 'Birytis': 8, 'Bisanthe': 12, 'Bizye': 35, 'Byzantion': 110, 'Chersonesus Thracica': 7, 'Dardanos': 20, 'Deultum': 5, 'Dikaia': 3, 'Dionysopolis': 8, 'Elaious': 8, 'Gambrion': 18, 'Gargara': 18, 'Gentinos': 1, 'Gergis': 9, 'Germe': 21, 'Hadrianeia': 13, 'Hadrianoi': 4, 'Hadrianopolis': 52, 'Hadrianotherai': 12, 'Hamaxitos': 1, 'Hephaistia': 32, 'Ilion': 42, 'Imbros': 17, 'Iolla': 6, 'Istros': 34, 'Kabyle': 2, 'Kallatis': 16, 'Kardia': 11, 'Kebren': 12, 'Kisthene': 1, 'Koila': 9, 'Kolonai': 0, 'Kypsela': 3, 'Kyzikos': 34, 'Lamponeia': 1, 'Lampsakos': 74, 'Lysimacheia': 14, 'Madytos': 1, 'Markianopolis': 9, 'Maroneia': 23, 'Mesembria': 14, 'Miletoupolis': 34, 'Myrina': 16, 'Neandreia': 9, 'Nikopolis ad Istrum': 13, 'Nikopolis ad Mestum': 0, 'Odessos': 22, 'Ophryneion': 15, 'Orthagoria': 1, 'Parion': 22, 'Pautalia': 1, 'Pergamon': 78, 'Perinthos': 93, 'Perperene': 16, 'Philippopolis': 32, 'Pionia': 18, 'Pitane': 31, 'Plakia': 3, 'Plotinopolis': 2, 'Poimanenon': 12, 'Priapos': 22, 'Prokonnesos': 16, 'Samothrake': 7, 'Selymbria': 15, 'Serdika': 15, 'Sestos': 16, 'Sigeion': 7, 'Skamandria': 9, 'Skepsis': 43, 'Tenedos': 3, 'Thasos': 19, 'Tomis': 2, 'Topeiros': 6, 'Traianopolis': 14, 'Unknown': 35, 'Zeleia': 2}
        
       
        for key in right_mints_RandomForrest:
            right_mints_RandomForrest[key] = mints_images[key] - right_mints_RandomForrest[key]
        for key in right_mints_LogisticRegression:
            right_mints_LogisticRegression[key] = mints_images[key] - right_mints_LogisticRegression[key]
    
          
        for key in mints_images:
            if right_mints_wmprediction[key] != 0:
                right_mints_wmprediction.update({key: right_mints_wmprediction[key]/mints_images[key]})
            else: 
                right_mints_wmprediction.update({key: 0})
            if right_mints_uwmprediction[key] != 0:
                right_mints_uwmprediction.update({key: right_mints_uwmprediction[key]/mints_images[key]})
            else: 
                right_mints_uwmprediction.update({key: 0})
            if  right_mints_pmprediction[key] != 0:
                right_mints_pmprediction.update({key: right_mints_pmprediction[key]/mints_images[key]})
            else: 
                right_mints_pmprediction.update({key: 0})
            right_mints_resnetobv.update({key: right_mints_resnetobv[key]/mints_images[key]})
            right_mints_resnetrev.update({key: right_mints_resnetrev[key]/mints_images[key]})
            right_mints_resnetboth.update({key: right_mints_resnetboth[key]/mints_images[key]})
            right_mints_vggobv.update({key: right_mints_vggobv[key]/mints_images[key]})
            right_mints_vggrev.update({key: right_mints_vggrev[key]/mints_images[key]})
            right_mints_vggboth.update({key: right_mints_vggboth[key]/mints_images[key]})
            right_mints_resnet101obv.update({key: right_mints_resnet101obv[key]/mints_images[key]})
            right_mints_resnet101rev.update({key: right_mints_resnet101rev[key]/mints_images[key]})
            right_mints_resnet101both.update({key: right_mints_resnet101both[key]/mints_images[key]})
            right_mints_RandomForrest.update({key: right_mints_RandomForrest[key]/mints_images[key]})
            right_mints_LogisticRegression.update({key:  right_mints_LogisticRegression[key]/mints_images[key]})
          
         
        mints = list(mints_images.keys())
        wmpred = list(right_mints_wmprediction.values())
        uwmpred = list(right_mints_uwmprediction.values())         
        pmpred = list(right_mints_pmprediction.values())
        resnetobv = list(right_mints_resnetobv.values())
        resnetrev = list(right_mints_resnetrev.values())
        resnetboth = list(right_mints_resnetboth.values())
        vggobv = list(right_mints_vggobv.values())
        vggrev = list(right_mints_vggrev.values())
        vggboth = list(right_mints_vggboth.values())
        resnet101obv = list(right_mints_resnet101obv.values())
        resnet101rev = list(right_mints_resnet101rev.values())
        resnet101both = list(right_mints_resnet101both.values())
        randomforrest = list(right_mints_RandomForrest.values())
        logreg = list(right_mints_LogisticRegression.values())
        x = np.arange(len(mints))

        width =0.2
        
        plt.figure(figsize=(18, 2))
        plt.bar(x-0.2, resnetobv, width, color='cyan')
        plt.bar(x, resnetrev, width, color='orange')
        plt.bar(x+0.2, resnetboth, width, color='green')
        plt.xticks(x, mints, rotation=90)
        plt.legend(["resnet50obv", "resnet50rev", "resnet50both"])
        
        fig1 = plt.gcf()
        plt.show()
        plt.draw()
        fig1.savefig('resultsresnet50.png', bbox_inches='tight')
    
        plt.figure(figsize=(18, 2))
        plt.bar(x-0.2, wmpred, width, color='cyan')
        plt.bar(x, uwmpred, width, color='orange')
        plt.bar(x+0.2, pmpred, width, color='green')
        plt.xticks(x, mints, rotation=90)
        plt.legend(["wmpred", "uwmpred", "pmpred"])
        
        fig1 = plt.gcf()
        plt.show()
        plt.draw()
        fig1.savefig('resultsvoting.png', bbox_inches='tight')
        
        plt.figure(figsize=(18, 2))
        plt.bar(x-0.2, resnet101obv, width, color='cyan')
        plt.bar(x, resnet101rev, width, color='orange')
        plt.bar(x+0.2, resnet101both, width, color='green')
        plt.xticks(x, mints, rotation=90)
        plt.legend(["resnet101obv", "resnet101rev", "resnet101both"])
        
        fig1 = plt.gcf()
        plt.show()
        plt.draw()
        fig1.savefig('resultsresnet101.png', bbox_inches='tight')
        
        
        plt.figure(figsize=(18, 2))
        plt.bar(x-0.2, vggobv, width, color='cyan')
        plt.bar(x, vggrev, width, color='orange')
        plt.bar(x+0.2, vggboth, width, color='green')
        plt.xticks(x, mints, rotation=90)
        plt.legend(["vgg16obv", "vgg16rev", "vgg16both"])
        
        fig1 = plt.gcf()
        plt.show()
        plt.draw()
        fig1.savefig('resultsvgg16.png', bbox_inches='tight')
        
        
        plt.figure(figsize=(18, 2))
        plt.bar(x-0.1, logreg, width, color='cyan')
        plt.bar(x+0.1, randomforrest, width, color='orange')
    
        plt.xticks(x, mints, rotation=90)
        plt.legend(["Stackig_lr", "Stacking_rf"])
        
        fig1 = plt.gcf()
        plt.show()
        plt.draw()
        fig1.savefig('vergleich_stacking.png', bbox_inches='tight')
        
        plt.figure(figsize=(18, 2))
        plt.bar(x-0.2, resnetboth, width, color='cyan')
        plt.bar(x, logreg, width, color='orange')
        plt.bar(x+0.2, wmpred, width, color='green')
        plt.xticks(x, mints, rotation=90)
        plt.legend(["ResNet50both", "Stacking_lr", "wmpred"])
        
        fig1 = plt.gcf()
        plt.show()
        plt.draw()
        fig1.savefig('vergleich.png', bbox_inches='tight')
     
        return right_mints_wmprediction, right_mints_uwmprediction, right_mints_pmprediction
        
        