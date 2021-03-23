import pickle
import os
from tqdm import tqdm
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

    def initial_disease_target_drug(self):
        """
        initial_disease_target_drug is used to obtain all TTD provided golden standard disease-target-drug relations which also existed in our constructed knowledge graph.
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


class obtain_experimental_data:


    ## 构造负样例
    def obtain_disease_target_drug_negative_examples(self,KG_file,experimental_disease_target_drug_file_positive,output_file):
        ## load KnowledgeGraph file
        file=open(KG_file,"rb")
        KG=pickle.load(file)
        file.close()
        output=open(output_file,"w+")
        ## construct positive training data
        entity_set={}
        drug_type_set={}
        target_type_set={}
        disease_type_set={}
        disease_target_drug=open(experimental_disease_target_drug_file_positive,"r")
        line=disease_target_drug.readline()
        while line:
            sline=line.split("\t")
            drug=sline[2].split(":")[1].strip("\n")
            disease=sline[0].split(":")[1].strip("\n")
            target=sline[1].split(":")[1].strip("\n")
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
            line=disease_target_drug.readline()
        negative_examples_count=0
        entity_list=list(KG)
        selected_examples={}
        while negative_examples_count < 5000000:
            if (negative_examples_count%50000 == 0):
                print(negative_examples_count)
            random_drug=random.choice(entity_list)
            random_target=random.choice(entity_list)
            random_disease=random.choice(entity_list)
            selected_example=random_drug+" "+random_target+" "+random_disease
            while selected_example in selected_examples:
                random_drug=random.choice(entity_list)
                random_target=random.choice(entity_list)
                random_disease=random.choice(entity_list)
                selected_example=random_drug+" "+random_target+" "+random_disease
            #while random_drug in selected_entities or random_target in selected_entities or random_disease in selected_entities:
            #    random_drug=random.choice(entity_list)
            #    random_target=random.choice(entity_list)
            #    random_disease=random.choice(entity_list)
            #selected_entities[random_drug]=1
            #selected_entities[random_target]=1
            #selected_entities[random_disease]=1
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
        print("Finished obtaining negative examples.")

            #def construct_negative_training_data(self,KG_file,experimental_disease_target_drug_file_negative,UMLS_type_dir,output_file):

    def construct_negative_training_data(self,KG_file,experimental_disease_target_drug_file_negative,UMLS_type_dir,output_file):
        ## load KnowledgeGraph file
        file=open(KG_file,"rb")
        KG=pickle.load(file)
        file.close()
        ## load UMLS_type_file
        file=open(UMLS_type_dir+"/predicate_vector.txt","rb")
        predicate_vector=pickle.load(file)
        file.close()
        file=open(UMLS_type_dir+"/entity_vector.txt","rb")
        entity_vector=pickle.load(file)
        file.close()
        output=open(output_file,"w+")
        ## construct positive training data
        disease_target_drug=open(experimental_disease_target_drug_file_negative,"r")
        line=disease_target_drug.readline()
        disease_target_drug_case_count=0
        while line:
            sline=line.split("\t")
            drug=sline[2].split(":")[1].strip("\n")
            disease=sline[0].split(":")[1].strip("\n")
            target=sline[1].split(":")[1].strip("\n")
            #print(drug+"\t"+target+"\t"+disease)
            self.construct_training_negative_data_based_one_dtd(KG,entity_vector,predicate_vector,drug,target,disease,output)
            disease_target_drug_case_count += 1
            line=disease_target_drug.readline()
        output.close()
        disease_target_drug.close()
        print(disease_target_drug_case_count)
        print("finished...")

    ## 该函数供construct_positive_training_data函数调用
    def construct_training_negative_data_based_one_dtd(self,KG,entity_vector,predicate_vector,drug,target,disease,output):

        ##
        # The struction of KG, KG是一个字典，其中结构如下：
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
        # 对于一种 drug-target-disease example 共有4种构造正例的可能，每一种可能都是一个长度为792的向量，分别是：
        # 在KG中
        # case 1: drug - PREDICATE_1 - target - PREDICATE_2 - disease  药物，靶标，疾病都直接关联。对于这样的例子也将其构造成702长度的vector,其形式为 drug - PREDICATE_1 - target - PREDICATE_1 - target - PREDICATE_2 - target - PREDICATE_2 - disease
        # case 2: drug - PREDICATE_1 - entity - PREDICATE_2 - target - PREDICATE_3 - disease 药物和靶标直接相关，药物和疾病都间接相关。
        # case 3: drug - PREDICATE_1 - target - PREDICATE_2 - entity - PREDICATE_3 - disease 药物和靶标间接相关，靶标和疾病直接相关。
        # case 4: drug - PREDICATE_1 - entity_1 - PREDICATE_2 - target- PREDICATE_3 - entity_2 - PREDICATE_4 - disease 药物，靶标，疾病都间接相关。
        # drug->target->
        if drug in KG:
            # case 1: drug - PREDICATE_1 - target - PREDICATE_2 - disease  药物，靶标，疾病都直接关联。
            # 对于这样的例子也将其构造成702长度的vector,其形式为:
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
                        # target part of vector: 第一个PREDICATE_1 - target - PREDICATE_1中的target，它的值从 REAL target中复制得到
                        vector[133+52:133+52+133] = vector[133+52+133+52:133+52+133+52+133]
                        # PREDICATE_1 part of vector：这第二个PREDICATE_1的值和第一个PREDICATE_1相同，从第一个PREDICATE_1那复制得到即可
                        vector[133+52+133:133+52+133+52] = vector[133:133+52]
                        # PREDICATE_2 part of vector
                        for predicate in KG[target]["OBJECTS"][disease]["PREDICATES"]:
                            vector[133+52+133+52+133+predicate_vector[predicate]] += KG[target]["OBJECTS"][disease]["PREDICATES"][predicate]
                        # target part of vector:PREDICATE_2 - target - PREDICATE_2这第二个target的值也和REAL target相同，从REAL target那复制得到即可
                        vector[133+52+133+52+133+52:133+52+133+52+133+52+133] = vector[133+52+133+52:133+52+133+52+133]
                        # PREDICATE_2 part of vector：这是求第二个PREDICATE_2的值，和第一个PREDICATE_2相同，复制过来即可
                        vector[133+52+133+52+133+52+133:133+52+133+52+133+52+133+52] = vector[133+52+133+52+133:133+52+133+52+133+52]
                        # disease part of vector
                        for umls_type in KG[target]["OBJECTS"][disease]["TYPES"]:
                            vector[133+52+133+52+133+52+133+52+entity_vector[umls_type]] += KG[target]["OBJECTS"][disease]["TYPES"][umls_type]
                        for umls_number in vector:
                            output.write(str(umls_number)+"\t")
                        output.write("0\n")
                        if len(vector) > 874:
                            print("case 1\t"+str(len(vector)))
                    # case 3: drug - PREDICATE_1 - target - PREDICATE_2 - entity - PREDICATE_3 - disease 药物和靶标间接相关，靶标和疾病直接相关。
                    # 得到的结果类型为 case 3: drug - PREDICATE_1 - target - PREDICATE_1 - target - PREDICATE_2 - entity - PREDICATE_3 - disease
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
                                    # --1: target作为object的时候
                                    for umls_type in KG[drug]["OBJECTS"][target]["TYPES"]:
                                        vector[133+52+133+52+entity_vector[umls_type]] += KG[drug]["OBJECTS"][target]["TYPES"][umls_type]
                                    # --2: target作为subject的时候
                                    for umls_type in KG[target]["TYPES"]:
                                        vector[133+52+133+52+entity_vector[umls_type]] += KG[target]["TYPES"][umls_type]
                                    # target of vector: PREDICATE_1 - target - PREDICATE_1这个target，其值和 REAL target相同，直接复制过来即可
                                    vector[133+52:133+52+133] = vector[133+52+133+52:133+52+133+52+133]
                                    # PREDICATE_1 of vector: 第二个PREDICATE_1，它的值和第一个PREDICATE_1相同，直接复制过来即可
                                    vector[133+52+133:133+52+133+52] = vector[133:133+52]
                                    # PREDICATE_2 of vector
                                    for predicate_2 in KG[target]["OBJECTS"][entity]["PREDICATES"]:
                                        vector[133+52+133+52+133+predicate_vector[predicate_2]] += KG[target]["OBJECTS"][entity]["PREDICATES"][predicate_2]
                                    # entity of vector: PREDICATE_2 - entity - PREDICATE_3 种的entity
                                    # -- 1 : entity作为object
                                    for umls_type in KG[target]["OBJECTS"][entity]["TYPES"]:
                                        vector[133+52+133+52+133+52+entity_vector[umls_type]] += KG[target]["OBJECTS"][entity]["TYPES"][umls_type]
                                    # --2 : entity 作为subject
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
            # case 2: drug - PREDICATE_1 - entity - PREDICATE_2 - target - PREDICATE_3 - disease 药物和靶标直接相关，药物和疾病都间接相关。
            # 得到的类型为： drug - PREDICATE_1 - entity - PREDICATE_2 - target - PREDICATE_3 - target - PREDICATE_3 - disease 药物和靶标直接相关，药物和疾病都间接相关。
            # 在图中存在的形态为 drug-entity-target-disease
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
                                    # entity part of vector: 这个entity也是既当做subject也当做object，因此所有的umls_type都要收集起来
                                    # --1 entity part of vector: entity当做object的时候
                                    for umls_type in KG[drug]["OBJECTS"][entity_1]["TYPES"]:
                                        vector[133+52+entity_vector[umls_type]] += KG[drug]["OBJECTS"][entity_1]["TYPES"][umls_type]
                                    # --2 entity part of vector: entity当做subject的时候
                                    for umls_type in KG[entity_1]["TYPES"]:
                                        vector[133+52+entity_vector[umls_type]] += KG[entity_1]["TYPES"][umls_type]
                                    # PREDICATE_2 part of vector
                                    for predicate_2 in KG[entity_1]["OBJECTS"][target]["PREDICATES"]:
                                        vector[133+52+133+predicate_vector[predicate_2]] += KG[entity_1]["OBJECTS"][target]["PREDICATES"][predicate_2]
                                    # the REAL target part of vector: target同样既作为subject也作为object
                                    # --1 target part of vector: target作为subject
                                    for umls_type in KG[entity_1]["OBJECTS"][target]["TYPES"]:
                                        vector[133+52+133+52+entity_vector[umls_type]] += KG[entity_1]["OBJECTS"][target]["TYPES"][umls_type]
                                    # --2 target part of vector: target作为object
                                    for umls_type in KG[target]["TYPES"]:
                                        vector[133+52+133+52+entity_vector[umls_type]] += KG[target]["TYPES"][umls_type]
                                    # PREDICATE_3 part of vector
                                    for predicate_3 in KG[target]["OBJECTS"][disease]["PREDICATES"]:
                                        vector[133+52+133+52+133+predicate_vector[predicate_3]] += KG[target]["OBJECTS"][disease]["PREDICATES"][predicate_3]
                                    # target part of vector: PREDICATE_3 - target - PREDICATE_3中的target,这个target和 REAL target值相同，直接复制到相应位置即可
                                    vector[133+52+133+52+133+52:133+52+133+52+133+52+133]=vector[133+52+133+52:133+52+133+52+133]
                                    # PREDICATE_3 part of vector:这是第二个PREDICATE_3，与第一个PREDICATE_3相同，直接复制过来就行
                                    vector[133+52+133+52+133+52+133:133+52+133+52+133+52+133+52] = vector[133+52+133+52+133:133+52+133+52+133+52]
                                    # disease part of vector
                                    for umls_type in KG[target]["OBJECTS"][disease]["TYPES"]:
                                        vector[133+52+133+52+133+52+133+52+entity_vector[umls_type]] += KG[target]["OBJECTS"][disease]["TYPES"][umls_type]
                                    for umls_number in vector:
                                        output.write(str(umls_number)+"\t")
                                    output.write("0\n")
                                    if len(vector) > 874:
                                        print("case 2\t"+str(len(vector)))
                                # case 4: drug - PREDICATE_1 - entity_1 - PREDICATE_2 - target- PREDICATE_3 - entity_2 - PREDICATE_4 - disease 药物，靶标，疾病都间接相关。
                                # 输出样例为：drug - PREDICATE_1 - entity_1 - PREDICATE_2 - target- PREDICATE_3 - entity_2 - PREDICATE_4 - disease
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
                                                # --1 : entity_1作为object
                                                for umls_type in KG[drug]["OBJECTS"][entity_1]["TYPES"]:
                                                    vector[133+52+entity_vector[umls_type]] += KG[drug]["OBJECTS"][entity_1]["TYPES"][umls_type]
                                                # --2 : entity_1作为subject
                                                for umls_type in KG[entity_1]["TYPES"]:
                                                    vector[133+52+entity_vector[umls_type]] += KG[entity_1]["TYPES"][umls_type]
                                                # PREDICATE_2 part of vector
                                                for predicate_2 in KG[entity_1]["OBJECTS"][target]["PREDICATES"]:
                                                    vector[133+52+133+predicate_vector[predicate_2]] += KG[entity_1]["OBJECTS"][target]["PREDICATES"][predicate_2]
                                                # target part of vector
                                                # --1 : target作为object
                                                for umls_type in KG[entity_1]["OBJECTS"][target]["TYPES"]:
                                                    vector[133+52+133+52+entity_vector[umls_type]] += KG[entity_1]["OBJECTS"][target]["TYPES"][umls_type]
                                                # --2 : target作为subject
                                                for umls_type in KG[target]["TYPES"]:
                                                    vector[133+52+133+52+entity_vector[umls_type]] += KG[target]["TYPES"][umls_type]
                                                # PREDICATE_3 part of vector
                                                for predicate_3 in KG[target]["OBJECTS"][entity_2]["PREDICATES"]:
                                                    vector[133+52+133+52+133+predicate_vector[predicate_3]] += KG[target]["OBJECTS"][entity_2]["PREDICATES"][predicate_3]
                                                # entity_2 part of vector
                                                # --1 : entity_2作为object
                                                for umls_type in KG[target]["OBJECTS"][entity_2]["TYPES"]:
                                                    vector[133+52+133+52+133+52+entity_vector[umls_type]] += KG[target]["OBJECTS"][entity_2]["TYPES"][umls_type]
                                                # --1 : entity_2作为subject
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

        # 如果drug直接在图中（G[drug]）,即drug是以object的形式存在图中的

    # 现在有了正负样例，需要将其分割成训练样本和测试样本：
    # 对于positive example（共有558个examples，其中很多疾病是重复的，如果统计疾病种类的话大概不会有特别多），将1/5拥有相同疾病的examples作为test data,其他做为training data，如果不足5个，则选一个。
    # 对于negative example （我构造了1000个examples，所有example中的实体都互相不重复），根据
    def split_positive_data(self,experimental_disease_target_drug_file_positive,output_dir):
        output_training=open(output_dir+"/experimental_training_positive_examples.txt","w+")
        output_test=open(output_dir+"/experimental_test_positive_examples.txt","w+")
        disease_target_drug=open(experimental_disease_target_drug_file_positive,"r")
        disease_dic={}
        line=disease_target_drug.readline()
        while line:
            sline=line.split("\t")
            drug=sline[2].split(":")[1].strip("\n")
            disease=sline[0].split(":")[1].strip("\n")
            target=sline[1].split(":")[1].strip("\n")
            if disease not in disease_dic:
                disease_dic[disease]={}
                disease_dic[disease]["Target:"+target+"\tDrug:"+drug]=1
            else:
                disease_dic[disease]["Target:"+target+"\tDrug:"+drug]=1
            line=disease_target_drug.readline()
        print("共有："+str(len(disease_dic))+"种疾病...")
        for disease in disease_dic:
            count=0
            positive_number=int(len(disease_dic[disease])/5)
            if positive_number == 0:
                positive_number += 1
            for target_drug in disease_dic[disease]:
                if count < positive_number:
                    output_test.write("Disease:"+disease+"\t"+target_drug+"\n")
                    count += 1
                else:
                    output_training.write("Disease:"+disease+"\t"+target_drug+"\n")
                    count += 1
        disease_target_drug.close()
        output_training.close()
        output_test.close()

    ## 构造药物-疾病测试集:该测试集用我们训练好的机器学习模型来针对一种疾病预测所有可能治疗它的药物，比如对于疾病disease_A,我们给出一个药物排名
    #  所包含的函数有： construct_hitat10_examples,
    #                   examine_existane_of_drug_disease_path_in_KG,
    #                   construct_test_vectors_of_examples,
    #  drug_B,drug_A,drug_C等。初步打算使用hit@10来对结果进行评价。
    #  构造该数据集的过程：1）从TTD中找到所有drug-disease在KG中存在的药物-疾病对儿作为验证正例（不能是drug-target-disease都在KG中，因为这部分已经作为训练集了）
    #                     2）针对每一个疾病disease_x,根据之前得到过的drug_types,target_types构造所有可能的drug-target-disease_x
    #                     3) 由构造的drug-target-disease_x和KG得到它的特征
    #                     4）使用训练好的机器学习模型对所有候选药物打分、排序、输出
    ##
    def construct_hitat10_examples(self,TTD_drug_disease_file,drug_target_disease_file,KG_file,output_dir):
        file=open(KG_file,"rb")
        KG=pickle.load(file)
        file.close()
        drug_target_disease_cases={}
        file=open(drug_target_disease_file,"r",encoding='utf-8', errors='replace')
        line=file.readline()
        while line:
            sline=line.split("\t")
            drug=sline[2].split(":")[1].strip("\n")
            disease=sline[0].split(":")[1].strip("\n")
            drug_target_disease_cases[drug+"\t"+disease]=1
            line=file.readline()
        file.close()
        output_disease_drug=open(output_dir+"/disease_drug_cases_hitat10_(1_2)_5.txt","w+")
        file=open(TTD_drug_disease_file,"r",encoding='utf-8', errors='replace')
        line=file.readline()
        positive_count=0
        negative_count=0
        while line:
            sline=line.split("\t")
            drug=self.process_en(sline[1])
            disease=self.process_en(sline[2])
            if drug+"\t"+disease not in drug_target_disease_cases:
                if self.examine_existance_of_drug_disease_path_in_KG(KG,drug,disease):
                    positive_count += 1
                    output_disease_drug.write("Drug:"+drug+"\t"+"Disease:"+disease+"\n")
                    if (positive_count%100 == 0):
                        print("找到的drug-disease为："+str(positive_count))
                else:
                    negative_count += 1
                    if (negative_count%100 == 0):
                        print("抛弃的drug-disease为："+str(negative_count))
            line=file.readline()
        output_disease_drug.close()
        file.close()
        print("finishied...")

    ## 构造要被测试的drug-target-disease例子的vector
    #  这里构造的vector
    #  最终只够早了42种disease所有可能的drug-target-disease的vector，因为硬盘满了（总共写入大概40多G）。所以实验验证42种疾病的hit@10
    def construct_test_vectors_of_examples(self,KG_file,UMLS_type_dir,drug_disease_file,candidate_drugs_file,candidate_targets_file,output_dir):
        ## load KnowledgeGraph file
        file=open(KG_file,"rb")
        KG=pickle.load(file)
        file.close()
        print("finished loading KnowledgeGraph...")
        ## load UMLS_type_file
        file=open(UMLS_type_dir+"/predicate_vector.txt","rb")
        predicate_vector=pickle.load(file)
        file.close()
        file=open(UMLS_type_dir+"/entity_vector.txt","rb")
        entity_vector=pickle.load(file)
        file.close()
        output_dtd=open(output_dir+"/test_disease_target_drug_(hitat10_examples).txt","w+")
        output_dtd_vector=open(output_dir+"/test_disease_target_drug_(hitat10_vectors).txt","w+")
        # 装载所有待测试的diseases 和 candidate drugs and targets
        diseases=[]
        candidate_drugs=[]
        candidate_targets=[]
        disease_drug={}
        file=open(drug_disease_file,"r",encoding='utf-8', errors='replace')
        line=file.readline()
        while line:
            sline=line.split("\t")
            disease=sline[1].split(":")[1].strip("\n")
            diseases.append(disease)
            drug=sline[0].split(":")[1].strip("\n")
            if disease not in disease_drug:
                disease_drug[disease]={}
                disease_drug[disease][drug]=1
            else:
                disease_drug[disease][drug]=1
            line=file.readline()
        file.close()
        file=open(candidate_drugs_file,"r",encoding='utf-8', errors='replace')
        line=file.readline()
        while line:
            sline=line.split("\t")
            drug=sline[1].strip("\n")
            candidate_drugs.append(drug)
            line=file.readline()
        file.close()
        #############################################
        #target_not_included={"4hppd":1,"b-cell_specific_transcription_factor":1,"hifph2":1,"cd29_antigen":1}
        #############################################
        file=open(candidate_targets_file,"r",encoding='utf-8', errors='replace')
        line=file.readline()
        while line:
            sline=line.split("\t")
            target=sline[1].strip("\n")
            ##不包含"4hppd"
            #if target not in target_not_included:
            #    candidate_targets.append(target)
            candidate_targets.append(target)
            line=file.readline()
        file.close()
        print("fished loading diseases, targets, drugs ...")
        #print("共有药物： "+str(len(candidate_drugs)))
        #print("共有靶标： "+str(len(candidate_targets)))
        #print("共有疾病： "+str(len(diseases)))
        count_cases=0
        candidate_diseases_100=[]
        while len(candidate_diseases_100) <= 100:
            candidate_diseases_100.append(random.choice(diseases))

        target_not_included={"4hppd":1,"b-cell_specific_transcription_factor":1,"hifph2":1,"cd29_antigen":1,"frizzled-7_receptor":1,"ebi1":1}
        for i in range(len(candidate_diseases_100)):
            print("这是第"+str(i)+"种疾病...")
            ## 为每种疾病候选100种药物，其中1种是已知治愈该病的药物，其他为随机选择的药物
            disease=candidate_diseases_100[i]
            candidate_drugs_100=[]
            for drug in disease_drug[disease]:
                candidate_drugs_100.append(drug)
            # 从candidate_drugs随机选择其他药物
            while len(candidate_drugs_100) <= 100:
                candidate_drugs_100.append(random.choice(candidate_drugs))
            ##
            total_cases_number=len(candidate_diseases_100)*len(candidate_targets)*len(candidate_drugs_100)
            percentage=int(total_cases_number/100)
            print("总共的药物-靶标-疾病的组合有："+str(total_cases_number))
            for j in range(len(candidate_targets)):
                target=candidate_targets[j]
                for k in range(len(candidate_drugs_100)):
                    if count_cases % 100 == 0:
                        print(str(int(count_cases/percentage))+" : "+str(count_cases)+" / "+str(total_cases_number))
                    count_cases += 1
                    drug=candidate_drugs_100[k]
                    if target not in target_not_included:
                        print(drug+"\t"+target+"\t"+disease)
                        cost_time=self.construct_test_data_vector_based_one_dtd(KG,entity_vector,predicate_vector,drug,target,disease,output_dtd,output_dtd_vector)
                        if cost_time>0.1:
                            target_not_included[target]=1
                            for target_excluded in target_not_included:
                                print(target_excluded,end="\t")
                            print()
                            print("被去掉的target共有："+str(len(target_not_included)))

        output_dtd.close()
        output_dtd_vector.close()
        print("被去掉的target共有："+str(len(target_not_included)))
        print(target_not_included)
        print("finished...")

    def construct_test_data_vector_based_one_dtd(self,KG,entity_vector,predicate_vector,drug,target,disease,output_dtd,output_vector):
        function_start_dtd=time.clock()
        ## 现在要建立的vector由 drug target disease组成，首先这三种实体肯定在KG中存在，所以不需要验证它们是否在KG中。
        # 分为4种情况：
        # case 1: drug-target-disease 直接相关
        # case 2: drug-entity-target-disease
        # case 3: drug-target-entity-disease
        # case 4: drug-entity-target-entity-disease
        ##

        # case 1: drug-target-disease 直接相关
        # case 1: drug - PREDICATE_1 - target - PREDICATE_2 - disease  药物，靶标，疾病都直接关联。
        # 对于这样的例子也将其构造成873长度的vector,其形式为:
        #  drug - PREDICATE_1 - target - PREDICATE_1 - target - PREDICATE_2 - target - PREDICATE_2 - disease
        if target in KG[drug]["OBJECTS"] and disease in KG[target]["OBJECTS"]:
            #print("case 1: drug-target-disease")
            #  drug part of vector
            start_dtd=time.clock()
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
            # target part of vector: 第一个PREDICATE_1 - target - PREDICATE_1中的target，它的值从 REAL target中复制得到
            vector[133+52:133+52+133] = vector[133+52+133+52:133+52+133+52+133]
            # PREDICATE_1 part of vector：这第二个PREDICATE_1的值和第一个PREDICATE_1相同，从第一个PREDICATE_1那复制得到即可
            vector[133+52+133:133+52+133+52] = vector[133:133+52]
            # PREDICATE_2 part of vector
            for predicate in KG[target]["OBJECTS"][disease]["PREDICATES"]:
                vector[133+52+133+52+133+predicate_vector[predicate]] += KG[target]["OBJECTS"][disease]["PREDICATES"][predicate]
            # target part of vector:PREDICATE_2 - target - PREDICATE_2这第二个target的值也和REAL target相同，从REAL target那复制得到即可
            vector[133+52+133+52+133+52:133+52+133+52+133+52+133] = vector[133+52+133+52:133+52+133+52+133]
            # PREDICATE_2 part of vector：这是求第二个PREDICATE_2的值，和第一个PREDICATE_2相同，复制过来即可
            vector[133+52+133+52+133+52+133:133+52+133+52+133+52+133+52] = vector[133+52+133+52+133:133+52+133+52+133+52]
            # disease part of vector
            for umls_type in KG[target]["OBJECTS"][disease]["TYPES"]:
                vector[133+52+133+52+133+52+133+52+entity_vector[umls_type]] += KG[target]["OBJECTS"][disease]["TYPES"][umls_type]
            output_dtd.write(disease+"\t"+target+"\t"+drug+"\tcase 1:drug-target-disease"+"\n")
            for umls_number in vector:
                output_vector.write(str(umls_number)+"\t")
            #之前的vector集合output_vector.write("1\n")是错误的，因为把标签写了进去
            output_vector.write("\n")
            end_dtd=time.clock()
            return end_dtd-start_dtd

        # case 2: drug-entity-target-disease
        # case 2: drug - PREDICATE_1 - entity - PREDICATE_2 - target - PREDICATE_3 - disease 药物和靶标直接相关，药物和疾病都间接相关。
        # 得到的类型为： drug - PREDICATE_1 - entity - PREDICATE_2 - target - PREDICATE_3 - target - PREDICATE_3 - disease 药物和靶标直接相关，药物和疾病都间接相关。
        # 在图中存在的形态为 drug-entity-target-disease
        elif disease in KG[target]["OBJECTS"] and self.examine_reachability_of_two_entities_in_KG(KG,drug,target):
            start_dtd=time.clock()
            for entity_1 in KG[drug]["OBJECTS"]:
                end_dtd=time.clock()
                if (end_dtd-start_dtd)>0.01:
                    return end_dtd-start_dtd
                if entity_1 in KG:
                    if target in KG[entity_1]["OBJECTS"]:
                        end_dtd=time.clock()
                        if (end_dtd-start_dtd)>0.01:
                            return end_dtd-start_dtd
                        #print("case 2: drug-entity-target-disease")
                        vector=[0]*873
                        # drug part of vector
                        for umls_type in KG[drug]["TYPES"]:
                            vector[entity_vector[umls_type]] += KG[drug]["TYPES"][umls_type]
                        # PREDICATE_1 part of vector
                        for predicate_1 in KG[drug]["OBJECTS"][entity_1]["PREDICATES"]:
                            vector[133+predicate_vector[predicate_1]] += KG[drug]["OBJECTS"][entity_1]["PREDICATES"][predicate_1]
                        # entity part of vector: 这个entity也是既当做subject也当做object，因此所有的umls_type都要收集起来
                        # --1 entity part of vector: entity当做object的时候
                        for umls_type in KG[drug]["OBJECTS"][entity_1]["TYPES"]:
                            vector[133+52+entity_vector[umls_type]] += KG[drug]["OBJECTS"][entity_1]["TYPES"][umls_type]
                        # --2 entity part of vector: entity当做subject的时候
                        for umls_type in KG[entity_1]["TYPES"]:
                            vector[133+52+entity_vector[umls_type]] += KG[entity_1]["TYPES"][umls_type]
                        # PREDICATE_2 part of vector
                        for predicate_2 in KG[entity_1]["OBJECTS"][target]["PREDICATES"]:
                            vector[133+52+133+predicate_vector[predicate_2]] += KG[entity_1]["OBJECTS"][target]["PREDICATES"][predicate_2]
                        # the REAL target part of vector: target同样既作为subject也作为object
                        # --1 target part of vector: target作为subject
                        for umls_type in KG[entity_1]["OBJECTS"][target]["TYPES"]:
                            vector[133+52+133+52+entity_vector[umls_type]] += KG[entity_1]["OBJECTS"][target]["TYPES"][umls_type]
                        # --2 target part of vector: target作为object
                        for umls_type in KG[target]["TYPES"]:
                            vector[133+52+133+52+entity_vector[umls_type]] += KG[target]["TYPES"][umls_type]
                        # PREDICATE_3 part of vector
                        for predicate_3 in KG[target]["OBJECTS"][disease]["PREDICATES"]:
                            vector[133+52+133+52+133+predicate_vector[predicate_3]] += KG[target]["OBJECTS"][disease]["PREDICATES"][predicate_3]
                        # target part of vector: PREDICATE_3 - target - PREDICATE_3中的target,这个target和 REAL target值相同，直接复制到相应位置即可
                        vector[133+52+133+52+133+52:133+52+133+52+133+52+133]=vector[133+52+133+52:133+52+133+52+133]
                        # PREDICATE_3 part of vector:这是第二个PREDICATE_3，与第一个PREDICATE_3相同，直接复制过来就行
                        vector[133+52+133+52+133+52+133:133+52+133+52+133+52+133+52] = vector[133+52+133+52+133:133+52+133+52+133+52]
                        # disease part of vector
                        for umls_type in KG[target]["OBJECTS"][disease]["TYPES"]:
                            vector[133+52+133+52+133+52+133+52+entity_vector[umls_type]] += KG[target]["OBJECTS"][disease]["TYPES"][umls_type]
                        for umls_number in vector:
                            output_vector.write(str(umls_number)+"\t")
                        output_vector.write("1\n")
                        output_dtd.write(disease+"\t"+target+"\t"+drug+"\tcase 2:drug-entity-target-disease\n")


        # case 3: drug-target-entity-disease
        # case 3: drug - PREDICATE_1 - target - PREDICATE_2 - entity - PREDICATE_3 - disease 药物和靶标间接相关，靶标和疾病直接相关。
        # 得到的结果类型为 case 3: drug - PREDICATE_1 - target - PREDICATE_1 - target - PREDICATE_2 - entity - PREDICATE_3 - disease
        elif target in KG[drug]["OBJECTS"] and self.examine_reachability_of_two_entities_in_KG(KG,target,disease):
            start_dtd=time.clock()
            for entity in KG[target]["OBJECTS"]:
                end_dtd=time.clock()
                if (end_dtd-start_dtd)>0.01:
                    return end_dtd-start_dtd
                if entity in KG:
                    if disease in KG[entity]["OBJECTS"]:
                        end_dtd=time.clock()
                        if (end_dtd-start_dtd) > 0.01:
                            return end_dtd-start_dtd
                        #print("case 3: drug-arget-entity-disease")
                        vector=[0]*873
                        # drug part of vector
                        for umls_type in KG[drug]["TYPES"]:
                            vector[entity_vector[umls_type]] += KG[drug]["TYPES"][umls_type]
                        # PREDICATE_1 part of vector
                        for predicate_1 in KG[drug]["OBJECTS"][target]["PREDICATES"]:
                            vector[133+predicate_vector[predicate_1]] += KG[drug]["OBJECTS"][target]["PREDICATES"][predicate_1]
                        # the REAL target of vector
                        # --1: target作为object的时候
                        for umls_type in KG[drug]["OBJECTS"][target]["TYPES"]:
                            vector[133+52+133+52+entity_vector[umls_type]] += KG[drug]["OBJECTS"][target]["TYPES"][umls_type]
                        # --2: target作为subject的时候
                        for umls_type in KG[target]["TYPES"]:
                            vector[133+52+133+52+entity_vector[umls_type]] += KG[target]["TYPES"][umls_type]
                        # target of vector: PREDICATE_1 - target - PREDICATE_1这个target，其值和 REAL target相同，直接复制过来即可
                        vector[133+52:133+52+133] = vector[133+52+133+52:133+52+133+52+133]
                        # PREDICATE_1 of vector: 第二个PREDICATE_1，它的值和第一个PREDICATE_1相同，直接复制过来即可
                        vector[133+52+133:133+52+133+52] = vector[133:133+52]
                        # PREDICATE_2 of vector
                        for predicate_2 in KG[target]["OBJECTS"][entity]["PREDICATES"]:
                            vector[133+52+133+52+133+predicate_vector[predicate_2]] += KG[target]["OBJECTS"][entity]["PREDICATES"][predicate_2]
                        # entity of vector: PREDICATE_2 - entity - PREDICATE_3 种的entity
                        # -- 1 : entity作为object
                        for umls_type in KG[target]["OBJECTS"][entity]["TYPES"]:
                            vector[133+52+133+52+133+52+entity_vector[umls_type]] += KG[target]["OBJECTS"][entity]["TYPES"][umls_type]
                        # --2 : entity 作为subject
                        for umls_type in KG[entity]["TYPES"]:
                            vector[133+52+133+52+133+52+entity_vector[umls_type]] += KG[entity]["TYPES"][umls_type]
                        # PREDICATE_3 of vector
                        for predicate_3 in KG[entity]["OBJECTS"][disease]["PREDICATES"]:
                            vector[133+52+133+52+133+52+133+predicate_vector[predicate_3]] += KG[entity]["OBJECTS"][disease]["PREDICATES"][predicate_3]
                        # disease of vector
                        for umls_type in KG[entity]["OBJECTS"][disease]["TYPES"]:
                            vector[133+52+133+52+133+52+133+52+entity_vector[umls_type]] += KG[entity]["OBJECTS"][disease]["TYPES"][umls_type]
                        for umls_number in vector:
                            output_vector.write(str(umls_number)+"\t")
                        output_vector.write("1\n")
                        output_dtd.write(disease+"\t"+target+"\t"+drug+"\tcase 3:drug-target-entity-disease\n")

        # case 4: drug-entity-target-entity-disease
        # case 4: drug - PREDICATE_1 - entity_1 - PREDICATE_2 - target- PREDICATE_3 - entity_2 - PREDICATE_4 - disease 药物，靶标，疾病都间接相关。
        # 输出样例为：drug - PREDICATE_1 - entity_1 - PREDICATE_2 - target- PREDICATE_3 - entity_2 - PREDICATE_4 - disease
        elif self.examine_existance_of_drug_disease_path_in_KG(KG,drug,target) and self.examine_existance_of_drug_disease_path_in_KG(KG,target,disease):
            start_dtd=time.clock()
            for entity_1 in KG[drug]["OBJECTS"]:
                end_dtd=time.clock()
                if (end_dtd-start_dtd)>0.01:
                    return end_dtd-start_dtd
                if entity_1 in KG:
                    if target in KG[entity_1]["OBJECTS"]:
                        if target in KG:
                            for entity_2 in KG[target]["OBJECTS"]:
                                end_dtd=time.clock()
                                if (end_dtd-start_dtd)>0.01:
                                    return end_dtd-start_dtd
                                if entity_2 in KG:
                                    if disease in KG[entity_2]["OBJECTS"]:
                                        vector=[0]*873
                                        #print("case 4: drug-entity-target-entity-disease")
                                        # drug part of vector
                                        for umls_type in KG[drug]["TYPES"]:
                                            vector[entity_vector[umls_type]] += KG[drug]["TYPES"][umls_type]
                                        # PREDICATE_1 part of vector
                                        for predicate_1 in KG[drug]["OBJECTS"][entity_1]["PREDICATES"]:
                                            vector[133+predicate_vector[predicate_1]] += KG[drug]["OBJECTS"][entity_1]["PREDICATES"][predicate_1]
                                        # entity_1 part of vector
                                        # --1 : entity_1作为object
                                        for umls_type in KG[drug]["OBJECTS"][entity_1]["TYPES"]:
                                            vector[133+52+entity_vector[umls_type]] += KG[drug]["OBJECTS"][entity_1]["TYPES"][umls_type]
                                        # --2 : entity_1作为subject
                                        for umls_type in KG[entity_1]["TYPES"]:
                                            vector[133+52+entity_vector[umls_type]] += KG[entity_1]["TYPES"][umls_type]
                                        # PREDICATE_2 part of vector
                                        for predicate_2 in KG[entity_1]["OBJECTS"][target]["PREDICATES"]:
                                            vector[133+52+133+predicate_vector[predicate_2]] += KG[entity_1]["OBJECTS"][target]["PREDICATES"][predicate_2]
                                        # target part of vector
                                        # --1 : target作为object
                                        for umls_type in KG[entity_1]["OBJECTS"][target]["TYPES"]:                                                vector[133+52+133+52+entity_vector[umls_type]] += KG[entity_1]["OBJECTS"][target]["TYPES"][umls_type]
                                            # --2 : target作为subject
                                        for umls_type in KG[target]["TYPES"]:
                                            vector[133+52+133+52+entity_vector[umls_type]] += KG[target]["TYPES"][umls_type]
                                        # PREDICATE_3 part of vector
                                        for predicate_3 in KG[target]["OBJECTS"][entity_2]["PREDICATES"]:
                                            vector[133+52+133+52+133+predicate_vector[predicate_3]] += KG[target]["OBJECTS"][entity_2]["PREDICATES"][predicate_3]
                                        # entity_2 part of vector
                                        # --1 : entity_2作为object
                                        for umls_type in KG[target]["OBJECTS"][entity_2]["TYPES"]:
                                            vector[133+52+133+52+133+52+entity_vector[umls_type]] += KG[target]["OBJECTS"][entity_2]["TYPES"][umls_type]
                                        # --1 : entity_2作为subject
                                        for umls_type in KG[entity_2]["TYPES"]:
                                            vector[133+52+133+52+133+52+entity_vector[umls_type]] += KG[entity_2]["TYPES"][umls_type]
                                        # PREDICATE_4 part of vector
                                        for predicate_4 in KG[entity_2]["OBJECTS"][disease]["PREDICATES"]:
                                            vector[133+52+133+52+133+52+133+predicate_vector[predicate_4]] += KG[entity_2]["OBJECTS"][disease]["PREDICATES"][predicate_4]
                                        # disease part of vector
                                        for umls_type in KG[entity_2]["OBJECTS"][disease]["TYPES"]:
                                            vector[133+52+133+52+133+52+133+52+entity_vector[umls_type]] += KG[entity_2]["OBJECTS"][disease]["TYPES"][umls_type]
                                        for umls_number in vector:
                                            output_vector.write(str(umls_number)+"\t")
                                        output_vector.write("1\n")
                                        output_dtd.write(disease+"\t"+target+"\t"+drug+"\tcase 4:drug-entity-target-entity-disease\n")
                                        end_dtd=time.clock()
                                        if (end_dtd-start_dtd)>0.01:
                                            return end_dtd-start_dtd
        return time.clock()-function_start_dtd

    ## 因为硬盘大小的关系，construct_test_vectors_of_examples只构造了43种疾病相关的例子，其中第43种疾病（cancer）并没有完成。
    # 因此下边extract_part_test_vectors_of_examples函数把完整的42种疾病提取出来。
    # 这个函数只运行一遍，因为运行完后将包含43（不完整）种疾病的原始文件删掉
    def extract_part_test_vectors_of_examples(self,example_dir):
        start=time.clock()
        example_file=open(example_dir+"/test_disease_target_drug_(hitat10_examples)_temp.txt","r")
        example_vector=open(example_dir+"/test_disease_target_drug_(hitat10_vectors)_temp.txt","r")
        output_example_file=open(example_dir+"/test_disease_target_drug_(hitat10_examples).txt","w+")
        output_example_vector=open(example_dir+"/test_disease_target_drug_(hitat10_vectors).txt","w+")
        line_example=example_file.readline()
        line_vector=example_vector.readline()
        disease=line_example.split("\t")[0]
        while disease != "cancer":
            output_example_file.write(line_example)
            output_example_vector.write(line_vector)
            line_example=example_file.readline()
            line_vector=example_vector.readline()
            disease=line_example.split("\t")[0]
        example_file.close()
        example_vector.close()
        output_example_file.close()
        output_example_vector.close()
        print("Spend "+str(time.clock()-start)+" finished")

    ## 从TTD的文件中收集drug和target可能出现的所有umls types
    #  obtain_umls_types_of_drugs和obtain_umls_types_of_targets这两个函数只用一次就可以了，得到的文件存储起来
    def obtain_umls_types_of_drugs(self,KG_file,drug_target_disease_file):
        umls_types_of_drug={}
        file=open(KG_file,"rb")
        KG=pickle.load(file)
        file.close()
        file=open(drug_target_disease_file)
        line=file.readline()
        while line:
            sline=line.split("\t")
            drug=sline[0].split(":")[1].strip("\n")
            for umls_type in KG[drug]["TYPES"]:
                if umls_type not in umls_types_of_drug:
                    umls_types_of_drug[umls_type]=1
                else:
                    umls_types_of_drug[umls_type] += KG[drug]["TYPES"][umls_type]
            line=file.readline()
        return umls_types_of_drug
    def obtain_umls_types_of_targets(self,KG_file,disease_target_drug_candidates_file,output_file):
        umls_types_of_target={}
        file=open(KG_file,"rb")
        KG=pickle.load(file)
        file.close()
        output=open(output_file,"w+")
        file=open(disease_target_drug_candidates_file,"r",encoding='utf-8', errors='replace')
        candidate_targets={}
        line=file.readline()
        while line:
            sline=line.split("\t")
            target=sline[1].split(":")[1].strip("\n")
            if target in KG:
                candidate_targets[target]=1
                for umls_type in KG[target]["TYPES"]:
                    if umls_type not in umls_types_of_target:
                        umls_types_of_target[umls_type]=1
                    else:
                        umls_types_of_target[umls_type] += KG[target]["TYPES"][umls_type]
            line=file.readline()
        for target in candidate_targets:
            output.write("Target:"+target+"\n")
        output.close()
        return umls_types_of_target
    ## 这个函数用来获取所有可能存在的药物和靶标：其中收集到的药物和靶标分成两部分：
    #  1. TTD中直接包含的drug或者target
    #  2. KG中符合上边得到的drug, target的umls type的实体
    # 结果：原有drug 3,283个，从KG补充后得到233,342个
    #       原有target 5,795个，从KG补充后得到294,099个
    # 最终：我决定实验中使用补充后的drug(471,323个)和原有的target（5,795个）
    # 原因：对于药物而言，各种化学物质都有可能会是潜在的药物；而对于target而言，还是借鉴TTD数据库总结好的靶标比较好
    def obtain_candidate_drugs_and_targets(self,KG_file,known_drug_file,known_target_file,umls_type_file_drug,umls_type_file_target,output_dir):
        file=open(KG_file,"rb")
        KG=pickle.load(file)
        file.close()
        candidate_drugs=[]
        candidate_drug_existance={}
        # step 1: 先将TTD中包含的药物（只是部分，不是TTD数据库中所有的drug，只是drug_disease_hitat10的所有drug）包含进来
        file=open(known_drug_file,"r")
        line=file.readline()
        while line:
            sline=line.split("\t")
            drug=sline[0].split(":")[1].strip("\n")
            if drug not in candidate_drug_existance:
                candidate_drug_existance[drug]=1
                candidate_drugs.append(drug)
            line=file.readline()
        file.close()
        print("现在候选药物的总数为："+str(len(candidate_drugs)))
        # step 2: 装载drug的所有umls type
        umls_type_of_drug={}
        file=open(umls_type_file_drug,"r")
        line=file.readline()
        while line:
            sline=line.split("\t")
            umls_type=sline[0]
            umls_type_of_drug[umls_type]=1
            line=file.readline()
        file.close()
        # step 3:从KG中补充所有符合umls_type_of_drug的实体作为候选可能的药物
        for entity in KG:
            for umls_type in KG[entity]["TYPES"]:
                if umls_type in umls_type_of_drug:
                    if entity not in candidate_drug_existance:
                        candidate_drug_existance[entity]=1
                        candidate_drugs.append(entity)
                        break
        print("现在所有候选药物的总数为："+str(len(candidate_drugs)))
        output_candidate_drug=open(output_dir+"/candate_drugs_7.txt","w+")
        for i in range(len(candidate_drugs)):
            output_candidate_drug.write(str(i)+"\t"+candidate_drugs[i]+"\n")
        output_candidate_drug.close()

        # 收集candidate targets -- step 1: 收集所有已知的targets
        candidate_target_existance={}
        candidate_targets=[]
        file=open(known_target_file,"r")
        line=file.readline()
        while line:
            sline=line.strip("\n").split(":")
            target=sline[1]
            if target not in candidate_target_existance:
                candidate_target_existance[target]=1
                candidate_targets.append(target)
            line=file.readline()
        file.close()
        print("现在候选target的总数为："+str(len(candidate_targets)))
        # step 2: 装载target的所有umls type
        umls_type_of_target={}
        file=open(umls_type_file_target,"r")
        line=file.readline()
        while line:
            sline=line.split("\t")
            umls_type=sline[0]
            umls_type_of_target[umls_type]=1
            line=file.readline()
        file.close()
        # step 3: 从KG中补充所有符合umls_type_of_target的实体作为候选可能的药物
        for entity in KG:
            for umls_type in KG[entity]["TYPES"]:
                if umls_type in umls_type_of_target:
                    if entity not in candidate_target_existance:
                        candidate_target_existance[entity]=1
                        candidate_targets.append(entity)
                        break
        print("现在所有候选target的总数为："+str(len(candidate_targets)))
        output_candidate_target=open(output_dir+"/candate_targets_7.txt","w+")
        for i in range(len(candidate_targets)):
            output_candidate_target.write(str(i)+"\t"+candidate_targets[i]+"\n")
        output_candidate_target.close()




## OBTAIN NEGATIVE DRUG-TARGET-DISEASE EXAMPLES
#KG_file="/home/bio/sangshengtian/paper_2_data/Knowledge_Graph_(1)_2/KnowledgeGraph.txt"
#experimental_disease_target_drug_file_positive="/home/bio/sangshengtian/paper_2_data/TTD_related_data_3/generatred_from_TTD/experimental_disease_target_drug_positive_(2)_3.txt"
#output_file="/home/bio/sangshengtian/paper_2_data/TTD_related_data_3/generatred_from_TTD/experimental_disease_target_drug_negative_(2)_3.txt"
#s=obtain_experimental_data()
#s.obtain_disease_target_drug_negative_examples(KG_file,experimental_disease_target_drug_file_positive,output_file)


## CONSTRUCT POSITIVE TRAINING DATA
#s=obtain_experimental_data()
#KG_file="/home/bio/sangshengtian/paper_2_data/Knowledge_Graph_(1)_2/KnowledgeGraph.txt"
#KG_file="/home/bio/sangshengtian/paper_2_data/Knowledge_Graph_(1)_2/KnowledgeGraph.txt"
#UMLS_type_dir="/home/bio/sangshengtian/paper_2_data/UMLS_types_4"
#the file of positive data for training
#experimental_disease_target_drug_file_positive="/home/bio/sangshengtian/paper_2_data/TTD_related_data_3/generatred_from_TTD/experimental_disease_target_drug_positive_(2)_3.txt"
#output_file_positive="/home/bio/sangshengtian/paper_2_data/training_and_test_data_(2_3_4)_5/all_positive_data.txt"
#the file of positive data for test
#experimental_disease_target_drug_file_negative="/home/bio/sangshengtian/paper_2_data/TTD_related_data_3/generatred_from_TTD/experimental_disease_target_drug_negative_(2)_3.txt"
#output_file_negative="/home/bio/sangshengtian/paper_2_data/training_and_test_data_(2_3_4)_5/all_negative_data.txt"
#s=obtain_experimental_data()
#s.construct_positive_training_data(KG_file,experimental_disease_target_drug_file_positive,UMLS_type_dir,output_file_positive)
#s.construct_negative_training_data(KG_file,experimental_disease_target_drug_file_negative,UMLS_type_dir,output_file_negative)

## SPLIT DATA INTO TRAINING AND TEST DATA
#experimental_disease_target_drug_file_positive="/home/bio/sangshengtian/paper_2_data/TTD_related_data_3/generatred_from_TTD/experimental_disease_target_drug_positive_(2)_3.txt"
#output_dir="/home/bio/sangshengtian/paper_2_data/TTD_related_data_3/generatred_from_TTD"
#s=obtain_experimental_data()
#s.split_positive_data(experimental_disease_target_drug_file_positive,output_dir)

## CONSTRUCT DRUG-DISEASE EXAMPLES:用hit@10来评价
#KG_file="/home/bio/sangshengtian/paper_2_data/Knowledge_Graph_(1)_2/KnowledgeGraph.txt"
#output_dir="/home/bio/sangshengtian/paper_2_data/TTD_related_data_3/generatred_from_TTD"
#disease_target_drug_cases_file="/home/bio/sangshengtian/paper_2_data/TTD_related_data_3/generatred_from_TTD/disease_target_drug_cases_(1)_2.txt"
#TTD_drug_disease_file="/home/bio/sangshengtian/paper_2_data/TTD_related_data_3/TTD/drug-disease_TTD2016.txt"
#s=obtain_experimental_data()
#s.construct_hitat10_examples(TTD_drug_disease_file,disease_target_drug_cases_file,KG_file,output_dir)


## OBTAIN UMLS TYPES OF DRUG AND TARGET
#KG_file="/home/bio/sangshengtian/paper_2_data/Knowledge_Graph_(1)_2/KnowledgeGraph.txt"
#drug_disease_file="/home/bio/sangshengtian/paper_2_data/TTD_related_data_3/generatred_from_TTD/disease_drug_cases_hitat10_(1_2)_5.txt"
#disease_target_drug_candidates_file="/home/bio/sangshengtian/paper_2_data/TTD_related_data_3/generatred_from_TTD/disease_target_drug_candidates_1.txt"
#output_file="/home/bio/sangshengtian/paper_2_data/TTD_related_data_3/generatred_from_TTD/candidate_target_(1_KG)_6.txt"
#s=obtain_experimental_data()
#umls_types_of_target=s.obtain_umls_types_of_targets(KG_file,disease_target_drug_candidates_file,output_file)
#print(len(umls_types_of_target))
#print(umls_types_of_target)
#output=open("/home/bio/sangshengtian/paper_2_data/TTD_related_data_3/generatred_from_TTD/umls_type_of_target_(1_KG)_6.txt","w+")
#ranked_types=sorted(umls_types_of_target, key=umls_types_of_target.get, reverse=True)
#for i in range(len(ranked_types)):
#    output.write(ranked_types[i]+"\t"+str(umls_types_of_target[ranked_types[i]])+"\n")
#output.write("\n")

## 获得所有的候选drug 和 target
#KG_file="/home/bio/sangshengtian/paper_2_data/Knowledge_Graph_(1)_2/KnowledgeGraph.txt"
#known_drug_file="/home/bio/sangshengtian/paper_2_data/TTD_related_data_3/generatred_from_TTD/disease_drug_cases_hitat10_(1_2)_5.txt"
#known_target_file="/home/bio/sangshengtian/paper_2_data/TTD_related_data_3/generatred_from_TTD/candidate_target_(1_KG)_6.txt"
#umls_type_file_drug="/home/bio/sangshengtian/paper_2_data/TTD_related_data_3/generatred_from_TTD/umls_type_of_drug_(5_KG)_6.txt"
#umls_type_file_target="/home/bio/sangshengtian/paper_2_data/TTD_related_data_3/generatred_from_TTD/umls_type_of_target_(1_KG)_6.txt"
#output_dir="/home/bio/sangshengtian/paper_2_data/TTD_related_data_3/generatred_from_TTD"
#s=obtain_experimental_data()
#s.obtain_candidate_drugs_and_targets(KG_file,known_drug_file,known_target_file,umls_type_file_drug,umls_type_file_target,output_dir)


# 构造测试集： 包括 drug-target-disease 的examples 和 其对应的vector
#KG_file="/home/bio/sangshengtian/paper_2_data/Knowledge_Graph_(1)_2/KnowledgeGraph.txt"
#UMLS_type_dir="/home/bio/sangshengtian/paper_2_data/UMLS_types_4"
#drug_disease_file="/home/bio/sangshengtian/paper_2_data/TTD_related_data_3/generatred_from_TTD/disease_drug_cases_hitat10_(1_2)_5.txt"
#candidate_drugs_file="/home/bio/sangshengtian/paper_2_data/TTD_related_data_3/generatred_from_TTD/candate_drugs_(5)_7.txt"
#candidate_targets_file="/home/bio/sangshengtian/paper_2_data/TTD_related_data_3/generatred_from_TTD/candidate_target_(1_KG)_7.txt"
#output_dir="/home/bio/sangshengtian/paper_2_data/training_and_test_data_(2_3_4)_5"
#s=obtain_experimental_data()
#s.construct_test_vectors_of_examples(KG_file,UMLS_type_dir,drug_disease_file,candidate_drugs_file,candidate_targets_file,output_dir)


##从最初产生的43种不完整的examples种得到完整的42种examples
#example_dir="/home/bio/sangshengtian/paper_2_data/training_and_test_data_(2_3_4)_5"
#s=obtain_experimental_data()
#s.extract_part_test_vectors_of_examples(example_dir)





if __name__ == "__main__":
    predication_dir = "./data/SemmedDB"
    TTD_dir = "./data/TTD"
    processed_dir = "./data/processed"
    s=Extract_Data(predication_dir,TTD_dir,processed_dir)
    #s.UMLS_type_vector()
    #s.drug_syndroms()
    #s.disease_target()
    #s.drug_disease()
    #s.initial_disease_target_drug()
    s.positive_training_data()

