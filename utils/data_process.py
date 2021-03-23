def process_en(sentence):
    sentence = sentence.lower().strip("\n. ")
    sentence_split = sentence.split(" ")
    new_sentence = sentence_split[0].strip(".")
    for split in sentence_split[1:]:
        split = split.strip(". ")
        new_sentence += "_"
        new_sentence += split
    if new_sentence[-1] == 's':
        new_sentence = new_sentence[:-1]
    return new_sentence

def examine_reachability_of_two_entities_in_KG(KG,entity_1,entity_2):
    for entity in KG[entity_1]["OBJECTS"]:
        if entity in KG:
            if entity_2 in KG[entity]["OBJECTS"]:
                return True
    return False

def examine_existance_of_drug_disease_path_in_KG(KG,drug,disease):
    if drug not in KG or disease not in KG:
        return False
    else:
        if disease in KG[drug]["OBJECTS"]:
            return True
        for entity in KG[drug]["OBJECTS"]:
            if entity in KG:
                if disease in KG[entity]["OBJECTS"]:
                    return True
        for entity_1 in KG[drug]["OBJECTS"]:
            if entity_1 in KG:
                for entity_2 in KG[entity_1]["OBJECTS"]:
                    if entity_2 in KG:
                        if disease in KG[entity_2]["OBJECTS"]:
                            return True
        for entity_1 in KG[drug]["OBJECTS"]:
            if entity_1 in KG:
                for entity_2 in KG[entity_1]["OBJECTS"]:
                    if entity_2 in KG:
                        for entity_3 in KG[entity_2]["OBJECTS"]:
                            if entity_3 in KG:
                                if disease in KG[entity_3]["OBJECTS"]:
                                    return True
    return False
