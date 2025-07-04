from unet_4d import build_4d_unet
from losses import *
from layer_util import *
from training_utils import *

def main():
    continue_training = 0
    data_path = 'resources/clean'
    debug = 0
    evaluate_only = 0

    model_name = '4DSEG-31'

    if continue_training:
        run = neptune.init_run(    
        project="force/4D-Seg",

        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2Nzk0ZmU4Zi00YjQxLTQ1YTMtOWU2Ny0xMjU0YjY5ZTU2OWUifQ==",
        with_id = model_name,
        source_files=["*.ipynb", "*.py","*.csv"])  
        
    else:
        run = neptune.init_run(
        project="force/4D-Seg",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2Nzk0ZmU4Zi00YjQxLTQ1YTMtOWU2Ny0xMjU0YjY5ZTU2OWUifQ==",
        source_files=["*.ipynb", "*.py","*.csv"])  
        model_name = list(run.__dict__.values())[-10]


    patients = sorted([pat.split('/')[-1].split('.nii.gz')[0] for pat in glob.glob(f'{data_path}/images/*.nii.gz')])
    
    if debug:
        patients = patients[:10]

    N_test = int(len(patients) * 0.2)
    N_val = int(len(patients) * 0.1)
    N_train = len(patients) - N_test - N_val


    train_patients = patients[:N_train]
    val_patients = patients[N_train:N_train + N_val]
    test_patients = patients[N_train + N_val:]

    run['N_train'] = len(train_patients)
    run['N_val'] = len(val_patients)
    run['N_test'] = len(test_patients)


    all_patients = {'test':test_patients,'train':train_patients,'val':val_patients}

    train_gen = CustomDataGen(patients = train_patients,
                              cohort = 'train', 
                              data_path = data_path).get_gen
    
    val_gen   = CustomDataGen(patients = val_patients, 
                              cohort = 'val', 
                              data_path = data_path).get_gen

    output_types = (tf.float32,tf.uint8)

    train_ds = tf.data.Dataset.from_generator(train_gen, output_types=output_types)
    val_ds = tf.data.Dataset.from_generator(val_gen, output_types=output_types)


    BATCH_SIZE = 1
    train_ds = train_ds.shuffle(int(len(train_patients)/2), seed = 42, reshuffle_each_iteration=True).batch(BATCH_SIZE).prefetch(-1)
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(-1)

    X, y = next(iter(train_ds))

    model = build_4d_unet(input_shape=(32, None, 128, 128, 1), num_classes=3)

    if continue_training:
        run[f'{model_name}'].download(f'models/{model_name}.h5')
        model.load_weights(f'models/{model_name}.h5')
        print(f'{model_name} loaded!')

    model.summary()

    model.compile(loss = focal_tversky_loss, optimizer=tf.keras.optimizers.Adam())


    es = EarlyStopping(monitor='loss', 
                   mode='min', 
                   verbose = 1, 
                   patience = 20)
    
    mc = ModelCheckpoint(f'models/{model_name}.h5',
                    save_best_only= True,
                        monitor='loss',
                        mode='min')
    
    neptune_callback = NeptuneCallback(run=run) 
    eval_every_epoch = CustomCallback(model_name = model_name, 
                                    all_patients = all_patients, 
                                    run = run, 
                                    data_path = data_path,
                                    output_types = output_types,
                                    save_every = 10)

    if evaluate_only:
        evaluate(model_name, all_patients, run, data_path, output_types)
    else:
        print('Training model...')
        model.fit(train_ds,
                validation_data = val_ds, 
                epochs=1 if debug else 500,
                callbacks=[es, mc,eval_every_epoch,neptune_callback])


    
    run.stop()

if __name__ == "__main__":
    main()