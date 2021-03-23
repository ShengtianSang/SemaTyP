import pickle
import os
from tqdm import tqdm
import random
from utils import data_process
from knowledge_graph import Construct_KG

class Extract_Data:
    """
    Extract_Data is used to construct training, validation and test data by extracting TTD related relation paths from knowledge graph.
    In this work, 7,144 $drug-target-disease$ are extracted from Therapeutic Target Database (TTD) as true cases (The details could be found
    in Supplementary Data 1 of our published paper "SemaTyP: A Knowledge Graph Based LiteratureMining Method for Drug Discovery").
    The $\ell$ is set to 4, $K$ is 133 and $M$ is 52. Based on the aforementioned construction of training data, 19,230 positive data are
    obtained. Each data is a length of 873 (133*5+52*4) vector. On the other side, for each $drug-target-disease$, we random replaced the
    drug, target and disease with other drug, target and disease. If the new triplet doesn't exist in TTD, then it is considered as a false
    example, which is denoted as $drug^{'}-target^{'}-disease^{'}$. Similarly, 19,230 negative training data is obtained from false cases.
    """
    def __init__(self,predication_dir, TTD_dir, processed_dir):
        self.predication_dir=predication_dir
        self.TTD_dir=TTD_dir
        self.output_dir=processed_dir

    def UMLS_type_vector(self):
        print("Construct the UMLS type vector...")
        with open(self.predication_dir+"/predications.txt", 'r') as f:
            entity_vector = {}
            predication_vector = {}

            for line in tqdm(f, total=sum(1 for _ in open(self.predication_dir+"/predications.txt", 'r'))):
                sline = line.split("\t")
                if sline[0] == "" or sline[1] == "":
                    continue
                predicate = sline[3]
                subject_type = sline[4]
                object_type = sline[5].strip("\n")

                if predicate not in predication_vector:
                    predication_vector[predicate] = len(predication_vector)

                if subject_type not in entity_vector:
                    entity_vector[subject_type] = len(entity_vector)

                if object_type not in entity_vector:
                    entity_vector[object_type] = len(entity_vector)

        pickle.dump(entity_vector, open(self.output_dir+"/entity_vector", "wb+"))
        pickle.dump(predication_vector, open(self.output_dir+"/predicate_vector", "wb+"))
        return entity_vector, predication_vector

    def drug_syndroms(self):
        """
        input: ./data/TTD/Synonyms.txt, which is downloaded from Therapeutic Target Database
        output: ./data/processed/drug_synonyms, which contains each drug and its corresponding treated synonys, the format of the drug_synonyms:
        1. drug_synonyms["drug_id"]
        2. drug_synonyms["drug_id"]["synonyms"]
        3. drug_synonyms[drug]["synonyms"][drug_synonym]=drug_id
        """
        print("Extracting the drug-syndroms information ...")
        with open(self.TTD_dir+"/Synonyms.txt", 'r') as f:
            drug_synonyms = {}
            for line in tqdm(f, total=sum(1 for _ in open(self.TTD_dir+"/Synonyms.txt", 'r'))):
                sline = line.split("\t")
                drug_id = sline[0].lower()
                drug_names = {}
                drug_names[data_process.process_en(sline[1].lower())] = drug_id
                synonyms = sline[2].lower().strip("\n").split(";")
                for s in synonyms:
                    synonym = data_process.process_en(s)
                    drug_names[synonym] = drug_id
                for drug_name in drug_names:
                    if drug_name not in drug_synonyms:
                        drug_synonyms[drug_name] = {}
                        drug_synonyms[drug_name]["drug_id"] = drug_names[drug_name]
                        drug_synonyms[drug_name]["synonyms"] = {}
                        for synonym in drug_names:
                            drug_synonyms[drug_name]["synonyms"][synonym] = drug_names[drug_name]
                        if drug_name in drug_synonyms[drug_name]["synonyms"]:
                            del drug_synonyms[drug_name]["synonyms"][drug_name]
                    else:
                        for synonym in drug_names:
                            drug_synonyms[drug_name]["synonyms"][synonym] = drug_names[drug_name]
                        if drug_name in drug_synonyms[drug_name]["synonyms"]:
                            del drug_synonyms[drug_name]["synonyms"][drug_name]

            pickle.dump(drug_synonyms, open(self.output_dir+"/drug_synonyms", "wb+"))

            return drug_synonyms

    def disease_target(self):
        """
        input: ./data/TTD/target-disease_TTD2016.txt file which downloaded from Therapeutic Target Database
        output: ./data/processed/disease_targets. The format of disease_target:
        1. disease_targets[Indication]={}
        2. disease_targets[Indication][target_1]=target_1_ID
        Indication is the disease name.
        """
        print("Extracting the disease-target relations ...")
        with open(self.TTD_dir+"/target-disease_TTD2016.txt", "r") as f:
            disease_targets={}
            next(f)
            for line in tqdm(f, total=sum(1 for _ in open(self.TTD_dir+"/target-disease_TTD2016.txt", "r"))):
                sline = line.split("\t")
                TTDTargetID = sline[0].lower()
                Target_Name = sline[1].lower()
                Indications = sline[2].lower().split(";")
                new_Target_Name = data_process.process_en(Target_Name)
                for Indication in Indications:
                    new_Indication = data_process.process_en(Indication)
                    if new_Indication not in disease_targets:
                        disease_targets[new_Indication] = {}
                        disease_targets[new_Indication][new_Target_Name] = TTDTargetID
                    else:
                        disease_targets[new_Indication][new_Target_Name] = TTDTargetID
            pickle.dump(disease_targets, open(self.output_dir+"/disease_targets","wb+"))

            return disease_targets

    def drug_disease(self):
        """
         input: ./data/TTD/target-disease_TTD2016.txt file which downloaded from Therapeutic Target Database
         output: 1) ./data/processed/disease_drug; 2)./data/processed/drug_disease

         1)disease_drug is a dictionary, the format is:
            disease_drug[disease_name]={} contains all drugs and corresponding ids which could treat the disease
            disease_drug[disease][drug]=drug_id

         2)drug_disease is a dictionary, the format is:
            drug_disease[drug]={} contain all diseases which could be treated by this drug,
            drug_disease[drug][disease]=drug_id
            The reason why we built the drug_disease dictionary is because there is no disease ID in the initial TTD drug-disease_TTD2016.txt file

        This Therapeutic Target Database currently contains 2,589 targets (including 397 successful, 723 clinical trial, and 1,469 research targets), and 31,614 drugs (including 2,071 approved, 9,528 clinical trial, 17,803 experimental drugs). 20,278 small molecules and 653 antisense drugs are with available structure or oligonucleotide sequence.
         """
        print("Extracting the drug-disease relations ...")
        disease_drug = {}
        drug_disease = {}
        with open(self.TTD_dir + "/drug-disease_TTD2016.txt", "r") as f:
            next(f)
            for line in tqdm(f, total=sum(1 for _ in open(self.TTD_dir + "/drug-disease_TTD2016.txt", "r"))):
                sline = line.split("\t")
                drug_id = sline[0].lower()
                drug = data_process.process_en(sline[1].lower().strip(" "))
                diseases = sline[2].lower().split(";")
                for disease in diseases:
                    disease_processed = data_process.process_en(disease)
                    if disease_processed not in disease_drug:
                        disease_drug[disease_processed] = {}
                        disease_drug[disease_processed][drug] = drug_id
                    else:
                        disease_drug[disease_processed][drug] = drug_id

                if drug not in drug_disease:
                    drug_disease[drug] = {}
                    for disease in diseases:
                        disease_processed = data_process.process_en(disease)
                        drug_disease[drug][disease_processed] = drug_id
                else:
                    for disease in diseases:
                        disease_processed = data_process.process_en(disease)
                        drug_disease[drug][disease_processed] = drug_id

        pickle.dump(disease_drug, open(self.output_dir+"/disease_drug", "wb+"))
        pickle.dump(drug_disease, open(self.output_dir+"/drug_disease", "wb+"))
        return drug_disease, disease_drug

    def positive_dtd_cases(self):
        """
        positive_dtd_cases is used to obtain all TTD provided golden standard disease-target-drug relations which also existed in our constructed knowledge graph.
        """
        entities_and_type = {}
        print("Collecting all semantic types of each entity ...")
        with open(self.predication_dir + "/predications.txt", "r") as f:
            for line in tqdm(f, total=sum(1 for _ in open(self.predication_dir + "/predications.txt", "r"))):
                sline = line.split("\t")
                if sline[0] == "" or sline[1] == "":
                    continue
                entity_1 = data_process.process_en(sline[0])
                entity_2 = data_process.process_en(sline[1])
                entity_1_type = sline[4]
                entity_2_type = sline[5].strip("\n")
                if entity_1 not in entities_and_type:
                    entities_and_type[entity_1] = {}
                    entities_and_type[entity_1][entity_1_type] = 1
                else:
                    entities_and_type[entity_1][entity_1_type] = 1

                if entity_2 not in entities_and_type:
                    entities_and_type[entity_2] = {}
                    entities_and_type[entity_2][entity_2_type] = 1
                else:
                    entities_and_type[entity_2][entity_2_type] = 1

        output=open(self.output_dir+"/experimental_disease_target_drug","w+")
        with open(self.TTD_dir + "/disease_target_drug_cases.txt", "r", encoding='utf-8', errors='replace') as f:
            for line in tqdm(f, total=sum(1 for _ in open(self.TTD_dir + "/disease_target_drug_cases.txt", "r"))):
                # Disease:sickle-cell_disease	Target:humanized_igg2	Drug:selg2
                sline = line.split("\t")
                disease = data_process.process_en(sline[0].split(":")[1])
                target = data_process.process_en(sline[1].split(":")[1])
                drug = data_process.process_en(sline[2].strip("\n").split(":")[1])
                if disease in entities_and_type and target in entities_and_type and drug in entities_and_type:
                    output.write("Disease:" + disease + "\tTarget:" + target + "\tDrug:" + drug + "\n")
        output.close()

    def positive_training_data(self):
        """
        positive_training_data is used to construct positive training data all each drug-target-disease cases from "experimental_disease_target_drug"
        """
        if os.path.exists(self.output_dir+"/KnowledgeGraph"):
            KG = pickle.load(open(self.output_dir+"/KnowledgeGraph", "rb"))
        else:
            constuct_KG= Construct_KG(self.predication_dir+"/predications.txt",self.output_dir+"/KnowledgeGraph")
            KG = constuct_KG.construct_KnowledgeGraph()

        if os.path.exists(self.output_dir+"/predicate_vector") and os.path.exists(self.output_dir+"/entity_vector"):
            entity_vector = pickle.load(open(self.output_dir+"/entity_vector", "rb"))
            predicate_vector = pickle.load(open(self.output_dir+"/predicate_vector", "rb"))
        else:
            entity_vector, predicate_vector = self.UMLS_type_vector()

        if not os.path.exists(self.output_dir+"/experimental_disease_target_drug"):
            self.initial_disease_target_drug()

        output = open(self.output_dir+"/all_positive_data", "w+")
        print("Constructing the positive training data ...")
        with open(self.output_dir + "/experimental_disease_target_drug", "r") as f:
            for line in tqdm(f, total=sum(1 for _ in open(self.output_dir + "/experimental_disease_target_drug", "r"))):
                sline = line.split("\t")
                drug = data_process.process_en(sline[2].split(":")[1].strip("\n"))
                disease = data_process.process_en(sline[0].split(":")[1].strip("\n"))
                target = data_process.process_en(sline[1].split(":")[1].strip("\n"))
                print("Constructing the %s\t%s\t%s\t relations ..." %(drug,target,disease))
                self.construct_training_positive_data_based_one_dtd(KG, entity_vector, predicate_vector, drug, target, disease, output)

        output.close()


    def construct_training_positive_data_based_one_dtd(self, KG, entity_vector, predicate_vector, drug, target,
                                                       disease, output):

        ##
        # The struction of KG, KG is a dictionary, and the following shows the format of the KG：
        # KG={
        #     subject:
        #            {"TYPES":{sysn:2,horm:1,htrf:3}
        #             "OBJECTS":
        #                       object_1:{
        #                                 "TYPES":{}
        #                                 "PREDICATES":{
        #                                              predicate_1:3,
        #                                              predicate_2:4
        #                      }}}}}
        ##
        ##
        # For one specific drug-target-disease example, there are 4 potential possible positive cases could be constructed, and each of the 4 is a vector of lenght 792:
        # In the KG,
        # case 1: drug - PREDICATE_1 - target - PREDICATE_2 - disease  Case 1 indicates the drug, target and disease are directly connected. For case 1, we construct a vector, the order of each eneity in the vector is drug - PREDICATE_1 - target - PREDICATE_1 - target - PREDICATE_2 - target - PREDICATE_2 - disease
        # case 2: drug - PREDICATE_1 - entity - PREDICATE_2 - target - PREDICATE_3 - disease Case 2 indicates that the target and disease are directly connected, while the drug and target are indirectly connected.
        # case 3: drug - PREDICATE_1 - target - PREDICATE_2 - entity - PREDICATE_3 - disease Case 3 and Case 4 are the other 2 situations.
        # case 4: drug - PREDICATE_1 - entity_1 - PREDICATE_2 - target- PREDICATE_3 - entity_2 - PREDICATE_4 - disease
        # drug->target->
        if drug in KG:
            # For case 1: drug - PREDICATE_1 - target - PREDICATE_2 - disease
            # We used a vector of lenght 873, the format is:
            # drug - PREDICATE_1 - target - PREDICATE_1 - target - PREDICATE_2 - target - PREDICATE_2 - disease
            if target in KG[drug]["OBJECTS"]:
                if target in KG:
                    if disease in KG[target]["OBJECTS"]:
                        ## construct vector for drug - PREDICATE_1 - target - PREDICATE_1 - target - PREDICATE_2 - target - PREDICATE_2 - disease
                        #  drug part of vector
                        vector = [0] * 873
                        for umls_type in KG[drug]["TYPES"]:
                            vector[entity_vector[umls_type]] += KG[drug]["TYPES"][umls_type]
                        #  PREDICATE_1 part of vector
                        for predicate in KG[drug]["OBJECTS"][target]["PREDICATES"]:
                            vector[133 + predicate_vector[predicate]] += KG[drug]["OBJECTS"][target]["PREDICATES"][
                                predicate]
                        #  the REAL target part of vector: in this part, the REAL target is both object (for drug) and subject （for disease）,so all the umls typs of target(as subject and object) should be collected in vector
                        #  -- 1 the REAL target part of vector: target as object
                        for umls_type in KG[drug]["OBJECTS"][target]["TYPES"]:
                            vector[133 + 52 + 133 + 52 + entity_vector[umls_type]] += \
                            KG[drug]["OBJECTS"][target]["TYPES"][umls_type]
                        #  -- 2 the REAL target part of vector: target as subject
                        for umls_type in KG[target]["TYPES"]:
                            vector[133 + 52 + 133 + 52 + entity_vector[umls_type]] += KG[target]["TYPES"][umls_type]
                        # target part of vector: The target of the first PREDICATE_1 - target - PREDICATE_1, the value if copied from the REAL target
                        vector[133 + 52:133 + 52 + 133] = vector[133 + 52 + 133 + 52:133 + 52 + 133 + 52 + 133]
                        # PREDICATE_1 part of vector: the valued of the second PREDICATE_1 is same as the first PREDICATE_1, which is copied from PREDICATE_1
                        vector[133 + 52 + 133:133 + 52 + 133 + 52] = vector[133:133 + 52]
                        # PREDICATE_2 part of vector
                        for predicate in KG[target]["OBJECTS"][disease]["PREDICATES"]:
                            vector[133 + 52 + 133 + 52 + 133 + predicate_vector[predicate]] += \
                            KG[target]["OBJECTS"][disease]["PREDICATES"][predicate]
                        # target part of vector:PREDICATE_2 - target - PREDICATE_2, the value of the second target is same as REAL target, which is copied from REAL target
                        vector[133 + 52 + 133 + 52 + 133 + 52:133 + 52 + 133 + 52 + 133 + 52 + 133] = vector[
                                                                                                      133 + 52 + 133 + 52:133 + 52 + 133 + 52 + 133]
                        # PREDICATE_2 part of vector:
                        vector[
                        133 + 52 + 133 + 52 + 133 + 52 + 133:133 + 52 + 133 + 52 + 133 + 52 + 133 + 52] = vector[
                                                                                                          133 + 52 + 133 + 52 + 133:133 + 52 + 133 + 52 + 133 + 52]
                        # disease part of vector
                        for umls_type in KG[target]["OBJECTS"][disease]["TYPES"]:
                            vector[133 + 52 + 133 + 52 + 133 + 52 + 133 + 52 + entity_vector[umls_type]] += \
                            KG[target]["OBJECTS"][disease]["TYPES"][umls_type]
                        for umls_number in vector:
                            output.write(str(umls_number) + "\t")
                        output.write("1\n")
                    # For case 3: drug - PREDICATE_1 - target - PREDICATE_2 - entity - PREDICATE_3 - disease
                    # The format of the result is drug - PREDICATE_1 - target - PREDICATE_1 - target - PREDICATE_2 - entity - PREDICATE_3 - disease
                    else:
                        for entity in KG[target]["OBJECTS"]:
                            if entity in KG:
                                if disease in KG[entity]["OBJECTS"]:
                                    vector = [0] * 873
                                    # drug part of vector
                                    for umls_type in KG[drug]["TYPES"]:
                                        vector[entity_vector[umls_type]] += KG[drug]["TYPES"][umls_type]
                                    # PREDICATE_1 part of vector
                                    for predicate_1 in KG[drug]["OBJECTS"][target]["PREDICATES"]:
                                        vector[133 + predicate_vector[predicate_1]] += \
                                        KG[drug]["OBJECTS"][target]["PREDICATES"][predicate_1]
                                    # the REAL target of vector
                                    # --1: The target is used as object
                                    for umls_type in KG[drug]["OBJECTS"][target]["TYPES"]:
                                        vector[133 + 52 + 133 + 52 + entity_vector[umls_type]] += \
                                        KG[drug]["OBJECTS"][target]["TYPES"][umls_type]
                                    # --2: The target is the subject
                                    for umls_type in KG[target]["TYPES"]:
                                        vector[133 + 52 + 133 + 52 + entity_vector[umls_type]] += \
                                        KG[target]["TYPES"][umls_type]
                                    # target of vector: PREDICATE_1 - target - PREDICATE_1, the value of the target if copied from REAL target
                                    vector[133 + 52:133 + 52 + 133] = vector[
                                                                      133 + 52 + 133 + 52:133 + 52 + 133 + 52 + 133]
                                    # PREDICATE_1 of vector: it's copied from the first PREDICATE_1
                                    vector[133 + 52 + 133:133 + 52 + 133 + 52] = vector[133:133 + 52]
                                    # PREDICATE_2 of vector
                                    for predicate_2 in KG[target]["OBJECTS"][entity]["PREDICATES"]:
                                        vector[133 + 52 + 133 + 52 + 133 + predicate_vector[predicate_2]] += \
                                        KG[target]["OBJECTS"][entity]["PREDICATES"][predicate_2]
                                    # entity of vector: PREDICATE_2 - entity - PREDICATE_3
                                    # -- 1 : entity is the object
                                    for umls_type in KG[target]["OBJECTS"][entity]["TYPES"]:
                                        vector[133 + 52 + 133 + 52 + 133 + 52 + entity_vector[umls_type]] += \
                                        KG[target]["OBJECTS"][entity]["TYPES"][umls_type]
                                    # --2 : entity is the subject
                                    for umls_type in KG[entity]["TYPES"]:
                                        vector[133 + 52 + 133 + 52 + 133 + 52 + entity_vector[umls_type]] += \
                                        KG[entity]["TYPES"][umls_type]
                                    # PREDICATE_3 of vector
                                    for predicate_3 in KG[entity]["OBJECTS"][disease]["PREDICATES"]:
                                        vector[
                                            133 + 52 + 133 + 52 + 133 + 52 + 133 + predicate_vector[predicate_3]] += \
                                        KG[entity]["OBJECTS"][disease]["PREDICATES"][predicate_3]
                                    # disease of vector
                                    for umls_type in KG[entity]["OBJECTS"][disease]["TYPES"]:
                                        vector[
                                            133 + 52 + 133 + 52 + 133 + 52 + 133 + 52 + entity_vector[umls_type]] += \
                                        KG[entity]["OBJECTS"][disease]["TYPES"][umls_type]
                                    for umls_number in vector:
                                        output.write(str(umls_number) + "\t")
                                    output.write("1\n")
            # For case 2: drug - PREDICATE_1 - entity - PREDICATE_2 - target - PREDICATE_3 - disease
            # The result format: drug - PREDICATE_1 - entity - PREDICATE_2 - target - PREDICATE_3 - target - PREDICATE_3 - disease
            else:
                for entity_1 in KG[drug]["OBJECTS"]:
                    if entity_1 in KG:
                        if target in KG[entity_1]["OBJECTS"]:
                            if target in KG:
                                if disease in KG[target]["OBJECTS"]:
                                    vector = [0] * 873
                                    # drug part of vector
                                    for umls_type in KG[drug]["TYPES"]:
                                        vector[entity_vector[umls_type]] += KG[drug]["TYPES"][umls_type]
                                    # PREDICATE_1 part of vector
                                    for predicate_1 in KG[drug]["OBJECTS"][entity_1]["PREDICATES"]:
                                        vector[133 + predicate_vector[predicate_1]] += \
                                        KG[drug]["OBJECTS"][entity_1]["PREDICATES"][predicate_1]
                                    # entity part of vector: This entity could be used as both subject and object, then all the umls_typy should be collected.
                                    # --1 entity part of vector: entity is the object
                                    for umls_type in KG[drug]["OBJECTS"][entity_1]["TYPES"]:
                                        vector[133 + 52 + entity_vector[umls_type]] += \
                                        KG[drug]["OBJECTS"][entity_1]["TYPES"][umls_type]
                                    # --2 entity part of vector: entity is the subject
                                    for umls_type in KG[entity_1]["TYPES"]:
                                        vector[133 + 52 + entity_vector[umls_type]] += KG[entity_1]["TYPES"][
                                            umls_type]
                                    # PREDICATE_2 part of vector
                                    for predicate_2 in KG[entity_1]["OBJECTS"][target]["PREDICATES"]:
                                        vector[133 + 52 + 133 + predicate_vector[predicate_2]] += \
                                        KG[entity_1]["OBJECTS"][target]["PREDICATES"][predicate_2]
                                    # the REAL target part of vector: target could be subject or object
                                    # --1 target part of vector: target is subject
                                    for umls_type in KG[entity_1]["OBJECTS"][target]["TYPES"]:
                                        vector[133 + 52 + 133 + 52 + entity_vector[umls_type]] += \
                                        KG[entity_1]["OBJECTS"][target]["TYPES"][umls_type]
                                    # --2 target part of vector: target is object
                                    for umls_type in KG[target]["TYPES"]:
                                        vector[133 + 52 + 133 + 52 + entity_vector[umls_type]] += \
                                        KG[target]["TYPES"][umls_type]
                                    # PREDICATE_3 part of vector
                                    for predicate_3 in KG[target]["OBJECTS"][disease]["PREDICATES"]:
                                        vector[133 + 52 + 133 + 52 + 133 + predicate_vector[predicate_3]] += \
                                        KG[target]["OBJECTS"][disease]["PREDICATES"][predicate_3]
                                    # target part of vector: PREDICATE_3 - target - PREDICATE_3, the target is same as REAL target
                                    vector[
                                    133 + 52 + 133 + 52 + 133 + 52:133 + 52 + 133 + 52 + 133 + 52 + 133] = vector[
                                                                                                           133 + 52 + 133 + 52:133 + 52 + 133 + 52 + 133]
                                    # PREDICATE_3 part of vector: the second PREDICATE_3 is same as the first PREDICATE_3
                                    vector[
                                    133 + 52 + 133 + 52 + 133 + 52 + 133:133 + 52 + 133 + 52 + 133 + 52 + 133 + 52] = vector[
                                                                                                                      133 + 52 + 133 + 52 + 133:133 + 52 + 133 + 52 + 133 + 52]
                                    # disease part of vector
                                    for umls_type in KG[target]["OBJECTS"][disease]["TYPES"]:
                                        vector[
                                            133 + 52 + 133 + 52 + 133 + 52 + 133 + 52 + entity_vector[umls_type]] += \
                                        KG[target]["OBJECTS"][disease]["TYPES"][umls_type]
                                    for umls_number in vector:
                                        output.write(str(umls_number) + "\t")
                                    output.write("1\n")
                                # For case 4: drug - PREDICATE_1 - entity_1 - PREDICATE_2 - target- PREDICATE_3 - entity_2 - PREDICATE_4 - disease
                                # The output format:drug - PREDICATE_1 - entity_1 - PREDICATE_2 - target- PREDICATE_3 - entity_2 - PREDICATE_4 - disease
                                else:
                                    for entity_2 in KG[target]["OBJECTS"]:
                                        if entity_2 in KG:
                                            if disease in KG[entity_2]["OBJECTS"]:
                                                vector = [0] * 873
                                                # drug part of vector
                                                for umls_type in KG[drug]["TYPES"]:
                                                    vector[entity_vector[umls_type]] += KG[drug]["TYPES"][umls_type]
                                                # PREDICATE_1 part of vector
                                                for predicate_1 in KG[drug]["OBJECTS"][entity_1]["PREDICATES"]:
                                                    vector[133 + predicate_vector[predicate_1]] += \
                                                    KG[drug]["OBJECTS"][entity_1]["PREDICATES"][predicate_1]
                                                # entity_1 part of vector
                                                # --1 : entity_1 is object
                                                for umls_type in KG[drug]["OBJECTS"][entity_1]["TYPES"]:
                                                    vector[133 + 52 + entity_vector[umls_type]] += \
                                                    KG[drug]["OBJECTS"][entity_1]["TYPES"][umls_type]
                                                # --2 : entity_1 is subject
                                                for umls_type in KG[entity_1]["TYPES"]:
                                                    vector[133 + 52 + entity_vector[umls_type]] += \
                                                    KG[entity_1]["TYPES"][umls_type]
                                                # PREDICATE_2 part of vector
                                                for predicate_2 in KG[entity_1]["OBJECTS"][target]["PREDICATES"]:
                                                    vector[133 + 52 + 133 + predicate_vector[predicate_2]] += \
                                                    KG[entity_1]["OBJECTS"][target]["PREDICATES"][predicate_2]
                                                # target part of vector
                                                # --1 : target is object
                                                for umls_type in KG[entity_1]["OBJECTS"][target]["TYPES"]:
                                                    vector[133 + 52 + 133 + 52 + entity_vector[umls_type]] += \
                                                    KG[entity_1]["OBJECTS"][target]["TYPES"][umls_type]
                                                # --2 : target is subject
                                                for umls_type in KG[target]["TYPES"]:
                                                    vector[133 + 52 + 133 + 52 + entity_vector[umls_type]] += \
                                                    KG[target]["TYPES"][umls_type]
                                                # PREDICATE_3 part of vector
                                                for predicate_3 in KG[target]["OBJECTS"][entity_2]["PREDICATES"]:
                                                    vector[133 + 52 + 133 + 52 + 133 + predicate_vector[
                                                        predicate_3]] += \
                                                    KG[target]["OBJECTS"][entity_2]["PREDICATES"][predicate_3]
                                                # entity_2 part of vector
                                                # --1 : entity_2 is object
                                                for umls_type in KG[target]["OBJECTS"][entity_2]["TYPES"]:
                                                    vector[133 + 52 + 133 + 52 + 133 + 52 + entity_vector[
                                                        umls_type]] += KG[target]["OBJECTS"][entity_2]["TYPES"][
                                                        umls_type]
                                                # --1 : entity_2 is subject
                                                for umls_type in KG[entity_2]["TYPES"]:
                                                    vector[133 + 52 + 133 + 52 + 133 + 52 + entity_vector[
                                                        umls_type]] += KG[entity_2]["TYPES"][umls_type]
                                                # PREDICATE_4 part of vector
                                                for predicate_4 in KG[entity_2]["OBJECTS"][disease]["PREDICATES"]:
                                                    vector[133 + 52 + 133 + 52 + 133 + 52 + 133 + predicate_vector[
                                                        predicate_4]] += \
                                                    KG[entity_2]["OBJECTS"][disease]["PREDICATES"][predicate_4]
                                                # disease part of vector
                                                for umls_type in KG[entity_2]["OBJECTS"][disease]["TYPES"]:
                                                    vector[
                                                        133 + 52 + 133 + 52 + 133 + 52 + 133 + 52 + entity_vector[
                                                            umls_type]] += \
                                                    KG[entity_2]["OBJECTS"][disease]["TYPES"][umls_type]
                                                for umls_number in vector:
                                                    output.write(str(umls_number) + "\t")
                                                output.write("1\n")
        else:
            print("NOT FOUND " + drug + "\t" + target + "\t" + disease)


    def negative_dtd_cases(self):
        """
        negative_dtd_cases is used to construct negative drug-target-disease associations.
        """
        if os.path.exists(self.output_dir + "/KnowledgeGraph"):
            KG = pickle.load(open(self.output_dir + "/KnowledgeGraph", "rb"))
        else:
            constuct_KG = Construct_KG(self.predication_dir + "/predications.txt", self.output_dir + "/KnowledgeGraph")
            KG = constuct_KG.construct_KnowledgeGraph()

        entity_set={}
        drug_type_set={}
        target_type_set={}
        disease_type_set={}
        output = open(self.output_dir + "/experimental_disease_target_drug_negative", "w+")
        print("Constructing the negative drug-target-disease cases ...")
        with open(self.output_dir + "/experimental_disease_target_drug", "r") as f:
            for line in tqdm(f, total=sum(1 for _ in open(self.output_dir + "/experimental_disease_target_drug", "r"))):
                sline=line.split("\t")
                drug=data_process.process_en(sline[2].split(":")[1].strip("\n"))
                disease=data_process.process_en(sline[0].split(":")[1].strip("\n"))
                target=data_process.process_en(sline[1].split(":")[1].strip("\n"))
                entity_set[drug]=1
                entity_set[target]=1
                entity_set[disease]=1
                if drug in KG:
                    for umls_type in KG[drug]["TYPES"]:
                        drug_type_set[umls_type]=1
                if target in KG:
                    for umls_type in KG[target]["TYPES"]:
                        target_type_set[umls_type]=1
                if disease in KG:
                    for umls_type in KG[disease]["TYPES"]:
                        disease_type_set[umls_type]=1

            negative_examples_count=0
            entity_list=list(KG)
            selected_examples={}
            while negative_examples_count < 5000000:
                random_drug=random.choice(entity_list)
                random_target=random.choice(entity_list)
                random_disease=random.choice(entity_list)
                selected_example=random_drug+" "+random_target+" "+random_disease
                while selected_example in selected_examples:
                    random_drug=random.choice(entity_list)
                    random_target=random.choice(entity_list)
                    random_disease=random.choice(entity_list)
                    selected_example=random_drug+" "+random_target+" "+random_disease

                if random_drug not in entity_set and random_target not in entity_set and random_disease not in entity_set:
                    drug_umls_type_exist=False
                    target_umls_type_exist=False
                    disease_umls_type_exist=False
                    for umls_type_of_drug in KG[random_drug]["TYPES"]:
                        if umls_type_of_drug in drug_type_set:
                            drug_umls_type_exist=True
                            break
                    for umls_type_of_target in  KG[random_target]["TYPES"]:
                        if umls_type_of_target in target_type_set:
                            target_umls_type_exist=True
                            break
                    for umls_type_of_disease in KG[random_disease]["TYPES"]:
                        if umls_type_of_disease in disease_type_set:
                            disease_umls_type_exist=True
                    if drug_umls_type_exist == target_umls_type_exist == disease_umls_type_exist == True:
                        output.write("Disease:"+random_disease+"\tTarget:"+random_target+"\tDrug:"+random_drug+"\n")
                        negative_examples_count += 1


    def negative_training_data(self):
        if os.path.exists(self.output_dir+"/KnowledgeGraph"):
            KG = pickle.load(open(self.output_dir+"/KnowledgeGraph", "rb"))
        else:
            constuct_KG= Construct_KG(self.predication_dir+"/predications.txt",self.output_dir+"/KnowledgeGraph")
            KG = constuct_KG.construct_KnowledgeGraph()

        if os.path.exists(self.output_dir+"/predicate_vector") and os.path.exists(self.output_dir+"/entity_vector"):
            entity_vector = pickle.load(open(self.output_dir+"/entity_vector", "rb"))
            predicate_vector = pickle.load(open(self.output_dir+"/predicate_vector", "rb"))
        else:
            entity_vector, predicate_vector = self.UMLS_type_vector()

        if not os.path.exists(self.output_dir + "/experimental_disease_target_drug_negative"):
            self.negative_dtd_cases()

        output = open(self.output_dir + "/all_negative_data", "w+")
        print("Constructing the negative training data ...")
        with open(self.output_dir + "/experimental_disease_target_drug_negative", "r") as f:
            for line in tqdm(f, total=sum(1 for _ in open(self.output_dir + "/experimental_disease_target_drug_negative", "r"))):
                sline=line.split("\t")
                drug=sline[2].split(":")[1].strip("\n")
                disease=sline[0].split(":")[1].strip("\n")
                target=sline[1].split(":")[1].strip("\n")
                self.construct_training_negative_data_based_one_dtd(KG,entity_vector,predicate_vector,drug,target,disease,output)
        output.close()

    def construct_training_negative_data_based_one_dtd(self,KG,entity_vector,predicate_vector,drug,target,disease,output):
        ##
        # The function is similar to construct_training_positive_data_based_one_dtd function.
        # The struction of KG：
        # KG={
        #     subject:
        #            {"TYPES":{sysn:2,horm:1,htrf:3}
        #             object_1:{
        #                      "TYPES":{}
        #                      predicate_1:3,
        #                      predicate_2:4
        #                      }}}}}
        ##
        ##
        if drug in KG:
            # For case 1: drug - PREDICATE_1 - target - PREDICATE_2 - disease
            #  drug - PREDICATE_1 - target - PREDICATE_1 - target - PREDICATE_2 - target - PREDICATE_2 - disease
            if target in KG[drug]["OBJECTS"]:
                if target in KG:
                    if disease in KG[target]["OBJECTS"]:
                        ## construct vector for drug - PREDICATE_1 - target - PREDICATE_1 - target - PREDICATE_2 - target - PREDICATE_2 - disease
                        #  drug part of vector
                        vector=[0]*873
                        for umls_type in KG[drug]["TYPES"]:
                            vector[entity_vector[umls_type]] += KG[drug]["TYPES"][umls_type]
                        #  PREDICATE_1 part of vector
                        for predicate in KG[drug]["OBJECTS"][target]["PREDICATES"]:
                            vector[133+predicate_vector[predicate]] += KG[drug]["OBJECTS"][target]["PREDICATES"][predicate]
                        #  the REAL target part of vector: in this part, the REAL target is both object (for drug) and subject （for disease）,so all the umls typs of target(as subject and object) should be collected in vector
                        #  -- 1 the REAL target part of vector: target as object
                        for umls_type in KG[drug]["OBJECTS"][target]["TYPES"]:
                            vector[133+52+133+52+entity_vector[umls_type]] += KG[drug]["OBJECTS"][target]["TYPES"][umls_type]
                        #  -- 2 the REAL target part of vector: target as subject
                        for umls_type in KG[target]["TYPES"]:
                            vector[133+52+133+52+entity_vector[umls_type]] += KG[target]["TYPES"][umls_type]
                        # target part of vector
                        vector[133+52:133+52+133] = vector[133+52+133+52:133+52+133+52+133]
                        # PREDICATE_1 part of vector
                        vector[133+52+133:133+52+133+52] = vector[133:133+52]
                        # PREDICATE_2 part of vector
                        for predicate in KG[target]["OBJECTS"][disease]["PREDICATES"]:
                            vector[133+52+133+52+133+predicate_vector[predicate]] += KG[target]["OBJECTS"][disease]["PREDICATES"][predicate]
                        # target part of vector:PREDICATE_2 - target - PREDICATE_2
                        vector[133+52+133+52+133+52:133+52+133+52+133+52+133] = vector[133+52+133+52:133+52+133+52+133]
                        # PREDICATE_2 part of vector
                        vector[133+52+133+52+133+52+133:133+52+133+52+133+52+133+52] = vector[133+52+133+52+133:133+52+133+52+133+52]
                        # disease part of vector
                        for umls_type in KG[target]["OBJECTS"][disease]["TYPES"]:
                            vector[133+52+133+52+133+52+133+52+entity_vector[umls_type]] += KG[target]["OBJECTS"][disease]["TYPES"][umls_type]
                        for umls_number in vector:
                            output.write(str(umls_number)+"\t")
                        output.write("0\n")

                    # For case 3: drug - PREDICATE_1 - target - PREDICATE_2 - entity - PREDICATE_3 - disease
                    # drug - PREDICATE_1 - target - PREDICATE_1 - target - PREDICATE_2 - entity - PREDICATE_3 - disease
                    else:
                        for entity in KG[target]["OBJECTS"]:
                            if entity in KG:
                                if disease in KG[entity]["OBJECTS"]:
                                    vector=[0]*873
                                    # drug part of vector
                                    for umls_type in KG[drug]["TYPES"]:
                                        vector[entity_vector[umls_type]] += KG[drug]["TYPES"][umls_type]
                                    # PREDICATE_1 part of vector
                                    for predicate_1 in KG[drug]["OBJECTS"][target]["PREDICATES"]:
                                        vector[133+predicate_vector[predicate_1]] += KG[drug]["OBJECTS"][target]["PREDICATES"][predicate_1]
                                    # the REAL target of vector
                                    # --1: target is object
                                    for umls_type in KG[drug]["OBJECTS"][target]["TYPES"]:
                                        vector[133+52+133+52+entity_vector[umls_type]] += KG[drug]["OBJECTS"][target]["TYPES"][umls_type]
                                    # --2: target is subject
                                    for umls_type in KG[target]["TYPES"]:
                                        vector[133+52+133+52+entity_vector[umls_type]] += KG[target]["TYPES"][umls_type]
                                    # target of vector: PREDICATE_1 - target - PREDICATE_1
                                    vector[133+52:133+52+133] = vector[133+52+133+52:133+52+133+52+133]
                                    # PREDICATE_1 of vector
                                    vector[133+52+133:133+52+133+52] = vector[133:133+52]
                                    # PREDICATE_2 of vector
                                    for predicate_2 in KG[target]["OBJECTS"][entity]["PREDICATES"]:
                                        vector[133+52+133+52+133+predicate_vector[predicate_2]] += KG[target]["OBJECTS"][entity]["PREDICATES"][predicate_2]
                                    # entity of vector: PREDICATE_2 - entity - PREDICATE_3
                                    # -- 1 : entity is object
                                    for umls_type in KG[target]["OBJECTS"][entity]["TYPES"]:
                                        vector[133+52+133+52+133+52+entity_vector[umls_type]] += KG[target]["OBJECTS"][entity]["TYPES"][umls_type]
                                    # --2 : entity is subject
                                    for umls_type in KG[entity]["TYPES"]:
                                        vector[133+52+133+52+133+52+entity_vector[umls_type]] += KG[entity]["TYPES"][umls_type]
                                    # PREDICATE_3 of vector
                                    for predicate_3 in KG[entity]["OBJECTS"][disease]["PREDICATES"]:
                                        vector[133+52+133+52+133+52+133+predicate_vector[predicate_3]] += KG[entity]["OBJECTS"][disease]["PREDICATES"][predicate_3]
                                    # disease of vector
                                    for umls_type in KG[entity]["OBJECTS"][disease]["TYPES"]:
                                        vector[133+52+133+52+133+52+133+52+entity_vector[umls_type]] += KG[entity]["OBJECTS"][disease]["TYPES"][umls_type]
                                    for umls_number in vector:
                                        output.write(str(umls_number)+"\t")
                                    output.write("0\n")
                                    if len(vector) > 874:
                                        print("case 3\t"+str(len(vector)))
            # case 2: drug - PREDICATE_1 - entity - PREDICATE_2 - target - PREDICATE_3 - disease
            else:
                for entity_1 in KG[drug]["OBJECTS"]:
                    if entity_1 in KG:
                        if target in KG[entity_1]["OBJECTS"]:
                            if target in KG:
                                if disease in KG[target]["OBJECTS"]:
                                    vector=[0]*873
                                    # drug part of vector
                                    for umls_type in KG[drug]["TYPES"]:
                                        vector[entity_vector[umls_type]] += KG[drug]["TYPES"][umls_type]
                                    # PREDICATE_1 part of vector
                                    for predicate_1 in KG[drug]["OBJECTS"][entity_1]["PREDICATES"]:
                                        vector[133+predicate_vector[predicate_1]] += KG[drug]["OBJECTS"][entity_1]["PREDICATES"][predicate_1]
                                    # entity part of vector
                                    # --1 entity part of vector: entity is object
                                    for umls_type in KG[drug]["OBJECTS"][entity_1]["TYPES"]:
                                        vector[133+52+entity_vector[umls_type]] += KG[drug]["OBJECTS"][entity_1]["TYPES"][umls_type]
                                    # --2 entity part of vector: entity is subject
                                    for umls_type in KG[entity_1]["TYPES"]:
                                        vector[133+52+entity_vector[umls_type]] += KG[entity_1]["TYPES"][umls_type]
                                    # PREDICATE_2 part of vector
                                    for predicate_2 in KG[entity_1]["OBJECTS"][target]["PREDICATES"]:
                                        vector[133+52+133+predicate_vector[predicate_2]] += KG[entity_1]["OBJECTS"][target]["PREDICATES"][predicate_2]
                                    # the REAL target part of vector
                                    # --1 target part of vector: target is subject
                                    for umls_type in KG[entity_1]["OBJECTS"][target]["TYPES"]:
                                        vector[133+52+133+52+entity_vector[umls_type]] += KG[entity_1]["OBJECTS"][target]["TYPES"][umls_type]
                                    # --2 target part of vector: target is object
                                    for umls_type in KG[target]["TYPES"]:
                                        vector[133+52+133+52+entity_vector[umls_type]] += KG[target]["TYPES"][umls_type]
                                    # PREDICATE_3 part of vector
                                    for predicate_3 in KG[target]["OBJECTS"][disease]["PREDICATES"]:
                                        vector[133+52+133+52+133+predicate_vector[predicate_3]] += KG[target]["OBJECTS"][disease]["PREDICATES"][predicate_3]
                                    # target part of vector: PREDICATE_3 - target - PREDICATE_3
                                    vector[133+52+133+52+133+52:133+52+133+52+133+52+133]=vector[133+52+133+52:133+52+133+52+133]
                                    # PREDICATE_3 part of vector:这是第二个PREDICATE_3
                                    vector[133+52+133+52+133+52+133:133+52+133+52+133+52+133+52] = vector[133+52+133+52+133:133+52+133+52+133+52]
                                    # disease part of vector
                                    for umls_type in KG[target]["OBJECTS"][disease]["TYPES"]:
                                        vector[133+52+133+52+133+52+133+52+entity_vector[umls_type]] += KG[target]["OBJECTS"][disease]["TYPES"][umls_type]
                                    for umls_number in vector:
                                        output.write(str(umls_number)+"\t")
                                    output.write("0\n")
                                    if len(vector) > 874:
                                        print("case 2\t"+str(len(vector)))
                                # For case 4: drug - PREDICATE_1 - entity_1 - PREDICATE_2 - target- PREDICATE_3 - entity_2 - PREDICATE_4 - disease
                                # Example: drug - PREDICATE_1 - entity_1 - PREDICATE_2 - target- PREDICATE_3 - entity_2 - PREDICATE_4 - disease
                                else:
                                    for entity_2 in KG[target]["OBJECTS"]:
                                        if entity_2 in KG:
                                            if disease in KG[entity_2]["OBJECTS"]:
                                                vector=[0]*873
                                                # drug part of vector
                                                for umls_type in KG[drug]["TYPES"]:
                                                    vector[entity_vector[umls_type]] += KG[drug]["TYPES"][umls_type]
                                                # PREDICATE_1 part of vector
                                                for predicate_1 in KG[drug]["OBJECTS"][entity_1]["PREDICATES"]:
                                                    vector[133+predicate_vector[predicate_1]] += KG[drug]["OBJECTS"][entity_1]["PREDICATES"][predicate_1]
                                                # entity_1 part of vector
                                                # --1 : entity_1 is object
                                                for umls_type in KG[drug]["OBJECTS"][entity_1]["TYPES"]:
                                                    vector[133+52+entity_vector[umls_type]] += KG[drug]["OBJECTS"][entity_1]["TYPES"][umls_type]
                                                # --2 : entity_1 is subject
                                                for umls_type in KG[entity_1]["TYPES"]:
                                                    vector[133+52+entity_vector[umls_type]] += KG[entity_1]["TYPES"][umls_type]
                                                # PREDICATE_2 part of vector
                                                for predicate_2 in KG[entity_1]["OBJECTS"][target]["PREDICATES"]:
                                                    vector[133+52+133+predicate_vector[predicate_2]] += KG[entity_1]["OBJECTS"][target]["PREDICATES"][predicate_2]
                                                # target part of vector
                                                # --1 : target is object
                                                for umls_type in KG[entity_1]["OBJECTS"][target]["TYPES"]:
                                                    vector[133+52+133+52+entity_vector[umls_type]] += KG[entity_1]["OBJECTS"][target]["TYPES"][umls_type]
                                                # --2 : target is subject
                                                for umls_type in KG[target]["TYPES"]:
                                                    vector[133+52+133+52+entity_vector[umls_type]] += KG[target]["TYPES"][umls_type]
                                                # PREDICATE_3 part of vector
                                                for predicate_3 in KG[target]["OBJECTS"][entity_2]["PREDICATES"]:
                                                    vector[133+52+133+52+133+predicate_vector[predicate_3]] += KG[target]["OBJECTS"][entity_2]["PREDICATES"][predicate_3]
                                                # entity_2 part of vector
                                                # --1 : entity_2 is object
                                                for umls_type in KG[target]["OBJECTS"][entity_2]["TYPES"]:
                                                    vector[133+52+133+52+133+52+entity_vector[umls_type]] += KG[target]["OBJECTS"][entity_2]["TYPES"][umls_type]
                                                # --1 : entity_2 is subject
                                                for umls_type in KG[entity_2]["TYPES"]:
                                                    vector[133+52+133+52+133+52+entity_vector[umls_type]] += KG[entity_2]["TYPES"][umls_type]
                                                # PREDICATE_4 part of vector
                                                for predicate_4 in KG[entity_2]["OBJECTS"][disease]["PREDICATES"]:
                                                    vector[133+52+133+52+133+52+133+predicate_vector[predicate_4]] += KG[entity_2]["OBJECTS"][disease]["PREDICATES"][predicate_4]
                                                # disease part of vector
                                                for umls_type in KG[entity_2]["OBJECTS"][disease]["TYPES"]:
                                                    vector[133+52+133+52+133+52+133+52+entity_vector[umls_type]] += KG[entity_2]["OBJECTS"][disease]["TYPES"][umls_type]
                                                for umls_number in vector:
                                                    output.write(str(umls_number)+"\t")
                                                output.write("0\n")
                                                if len(vector) > 874:
                                                    print("case 4\t"+str(len(vector)))

    def construct_all_data(self):
        if not os.path.exists(self.output_dir + "/predicate_vector"):
            self.UMLS_type_vector()

        if not os.path.exists(self.output_dir + "/drug_synonyms"):
            self.drug_syndroms()

        if not os.path.exists(self.output_dir + "/disease_targets"):
            self.disease_target()

        if not os.path.exists(self.output_dir + "/drug_disease"):
            self.drug_disease()

        if not os.path.exists(self.output_dir + "/experimental_disease_target_drug"):
            self.positive_dtd_cases()

        if not os.path.exists(self.output_dir + "/all_positive_data"):
            self.positive_training_data()

        if not os.path.exists(self.output_dir + "/experimental_disease_target_drug_negative"):
            self.negative_dtd_cases()

        if not os.path.exists(self.output_dir + "/all_negative_data"):
            self.negative_training_data()

if __name__ == "__main__":
    predication_dir = "./data/SemmedDB"
    TTD_dir = "./data/TTD"
    processed_dir = "./data/processed"
    s=Extract_Data(predication_dir,TTD_dir,processed_dir)
    s.construct_all_data()

