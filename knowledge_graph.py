import pickle
from tqdm import tqdm
from utils import data_process

class Construct_KG:
    """
    Construct_KG is used to construct a semantic knowledge graph - called SemKG - from predicted relations extracted from PubMed abstracts.
    The knowledge graph is a multi-relational graph composed of entities as nodes and relations as different types of edges.
    In the SemKG, $E=\{e_1,e_2,...,e_N\}$ denotes the set of n entities, $R=\{r_1,r_2,...,r_M\}$ denote the set of relations between entities
    and $T=\{t_1,t_2,...,t_K\}$ denote semantic type of entities. The elements of R and T are all from the UMLS semantic network.
    The edge between entities $e_i$ and $e_j$ is weighted by the number of predications that have been extracted.
    Besides, the attribute of edge includes the abstracts' PubMed ID (pmid) from where the predications are extracted.
    A prototype example of the SemKG is illustrated in figure 1 of our published paper "SemaTyP: a knowledge graph based literature mining method for drug discovery"
    """
    def __init__(self,predication_file,output_file):
        """
        predication_file contains the relations extracted from PubMed abstracts, each line of predication_file indicates one relation, the format is exampled as:
        1) focal hand dystonia	patients	with	PROCESS_OF	dsyn	podg
        2) hand	focal hand dystonia	hand dystonia	PROCESS_OF	dsyn	podg
        The entities in the relation represent: entity1, entity2, the relation between the two enetities, the type of entity1, the type of entity2.
        """
        self.predication_file=predication_file
        self.output_file=output_file

    def construct_KnowledgeGraph(self):
        KG={}
        print("Construct the knowledge graph using the relations extracted from PubMed abstracts...")
        with open(self.predication_file, 'r') as f:
            for line in tqdm(f, total=sum(1 for line in open(self.predication_file, 'r'))):
                #hand	focal hand dystonia	hand dystonia	PROCESS_OF	dsyn	podg
                sline=line.split("\t")
                if sline[0] == "" or sline[1] == "":
                    continue
                subject=data_process.process_en(sline[0])
                object=data_process.process_en(sline[1])
                predicate=sline[3]
                subject_type=sline[4]
                object_type=sline[5].strip("\n")
                if subject not in KG:
                    KG[subject]={}
                    KG[subject]["TYPES"]={}
                    KG[subject]["TYPES"][subject_type]=1
                    KG[subject]["OBJECTS"]={}
                    KG[subject]["OBJECTS"][object]={}
                    KG[subject]["OBJECTS"][object]["TYPES"]={}
                    KG[subject]["OBJECTS"][object]["TYPES"][object_type]=1
                    KG[subject]["OBJECTS"][object]["PREDICATES"]={}
                    KG[subject]["OBJECTS"][object]["PREDICATES"][predicate]=1
                elif object not in KG[subject]["OBJECTS"]:
                    if subject_type not in KG[subject]["TYPES"]:
                        KG[subject]["TYPES"][subject_type]=1
                    else:
                        KG[subject]["TYPES"][subject_type] += 1
                    KG[subject]["OBJECTS"][object]={}
                    KG[subject]["OBJECTS"][object]["TYPES"]={}
                    KG[subject]["OBJECTS"][object]["TYPES"][object_type]=1
                    KG[subject]["OBJECTS"][object]["PREDICATES"]={}
                    KG[subject]["OBJECTS"][object]["PREDICATES"][predicate]=1
                else:
                    if subject_type not in KG[subject]["TYPES"]:
                        KG[subject]["TYPES"][subject_type]=1
                    else:
                        KG[subject]["TYPES"][subject_type] += 1
                    if object_type not in KG[subject]["OBJECTS"][object]["TYPES"]:
                        KG[subject]["OBJECTS"][object]["TYPES"][object_type]=1
                    else:
                        KG[subject]["OBJECTS"][object]["TYPES"][object_type] += 1
                    if predicate not in KG[subject]["OBJECTS"][object]["PREDICATES"]:
                        KG[subject]["OBJECTS"][object]["PREDICATES"][predicate]=1
                    else:
                        KG[subject]["OBJECTS"][object]["PREDICATES"][predicate] += 1

        pickle.dump(KG,open(self.output_file,"wb+"))
        return KG


if __name__ == "__main__":
    predication_file="./data/SemmedDB/predications.txt"
    output_file="./data/processed/KnowledgeGraph"
    s=Construct_KG(predication_file,output_file)
    s.construct_KnowledgeGraph()