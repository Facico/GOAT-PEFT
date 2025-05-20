from peta.tasks.cv import get_dataset

for task in [
"Cars"     ,
"DTD"      ,
"EuroSAT"  ,
"GTSRB"    ,
"MNIST"    ,
"RESISC45" ,
"SUN397"   ,
"SVHN"   
]:
    dataset = get_dataset(
        task + 'Val',  # xxVal is the train set !
        None, 
        location='../dataset/~data', 
        batch_size=1
    )
    test_dataset = get_dataset(
        task,  # xx is the test set !
        None, 
        location='../dataset/~data', 
        batch_size=1     
    )