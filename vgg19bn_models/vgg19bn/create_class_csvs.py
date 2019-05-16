import os

def create_csv(bone_type):
    print(bone_type)
    directory_file = os.listdir("../data/MURA-v1.1/data2/train/0/")
    print(len(directory_file))
    elbows = []
    count = 0
    for i in directory_file:

        if count % 1000 == 0:
            print(count)
        count += 1
        if bone_type in i:
            elbows.append("train/0/" + i)


    directory_file = os.listdir("../data/MURA-v1.1/data2/train/1/")
    print(len(directory_file))
    elbows2 = []
    count = 0
    for i in directory_file:

        if count % 1000 == 0:
            print(count)
        count += 1
        if bone_type in i:
            elbows2.append("train/1/" + i)

    directory_file = os.listdir("../data/MURA-v1.1/data2/valid/0/")
    print(len(directory_file))
    val_elbows = []
    count = 0
    for i in directory_file:

        if count % 1000 == 0:
            print(count)
        count += 1
        if bone_type in i:
            val_elbows.append("valid/0/" + i)


    directory_file = os.listdir("../data/MURA-v1.1/data2/valid/1/")
    print(len(directory_file))
    val_elbows2 = []
    count = 0
    for i in directory_file:

        if count % 1000 == 0:
            print(count)
        count += 1
        if bone_type in i:
            val_elbows2.append("valid/1/" + i)


    values = [0]*len(elbows) + [1]*len(elbows2) + [0]*len(val_elbows) + [1]*len(val_elbows2)
    valid = [False]*(len(elbows)+len(elbows2)) + [True]*(len(val_elbows) + len(val_elbows2))

    import pandas as pd

    elbows_df = pd.DataFrame({"name": elbows+elbows2+val_elbows+val_elbows2, "value": values, "is_valid": valid})

    elbows_df.to_csv("all_"+ bone_type + ".csv", index = False)
"""   
bone_types = ["ELBOW", "FINGER", "FOREARM", "HAND", "HUMERUS", "SHOULDER", "WRIST"]
for i in bone_types:
    create_csv(i)
"""    
def create_all_csv():
    bone_types = ["ELBOW", "FINGER", "FOREARM", "HAND", "HUMERUS", "SHOULDER", "WRIST"]
    directory_file = os.listdir("../data/MURA-v1.1/data2/train/0/")
    print(len(directory_file))
    elbows = []
    values = []
    valid = []
    count = 0
    for i in directory_file:

        if count % 1000 == 0:
            print(count)
        count += 1
        for bone in range(len(bone_types)):
            if bone_types[bone] in i:
                elbows.append("train/0/" + i)
                values.append(bone)
                valid.append(False)
        


    directory_file = os.listdir("../data/MURA-v1.1/data2/train/1/")
    print(len(directory_file))

    count = 0
    for i in directory_file:

        if count % 1000 == 0:
            print(count)
        count += 1
        for bone in range(len(bone_types)):
            if bone_types[bone] in i:
                elbows.append("train/1/" + i)
                values.append(bone)
                valid.append(False)

    directory_file = os.listdir("../data/MURA-v1.1/data2/valid/0/")
    print(len(directory_file))
    count = 0
    for i in directory_file:

        if count % 1000 == 0:
            print(count)
        count += 1
        for bone in range(len(bone_types)):
            if bone_types[bone] in i:
                elbows.append("valid/0/" + i)
                values.append(bone)
                valid.append(True)



    directory_file = os.listdir("../data/MURA-v1.1/data2/valid/1/")
    print(len(directory_file))

    count = 0
    for i in directory_file:

        if count % 1000 == 0:
            print(count)
        count += 1
        for bone in range(len(bone_types)):
            if bone_types[bone] in i:
                elbows.append("valid/1/" + i)
                values.append(bone)
                valid.append(True)



    # values = [0]*len(elbows) + [1]*len(elbows2) + [0]*len(val_elbows) + [1]*len(val_elbows2)
    # valid = [False]*(len(elbows)+len(elbows2)) + [True]*(len(val_elbows) + len(val_elbows2))

    import pandas as pd

    elbows_df = pd.DataFrame({"name": elbows, "value": values, "is_valid": valid})

    elbows_df.to_csv("all_bones.csv", index = False)
    
create_all_csv()