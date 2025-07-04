from training_utils import *
from losses import *
from layer_util import *

def main():
    data_path = 'resources/clean_subset'
    patients = sorted([pat.split('/')[-1].split('.nii.gz')[0] for pat in glob.glob(f'{data_path}/images/*.nii.gz')])

    N_test = int(len(patients) * 0.2)
    N_val = int(len(patients) * 0.1)
    N_train = len(patients) - N_test - N_val


    train_patients = patients[:N_train]
    val_patients = patients[N_train:N_train + N_val]
    test_patients = patients[N_train + N_val:]

    all_patients = {'test':test_patients,'train':train_patients,'val':val_patients}


    continue_training = 0
    model_name = '4DSEG-21'

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

    run['N_train'] = len(train_patients)
    run['N_val'] = len(val_patients)
    run['N_test'] = len(test_patients)

    output_channels = 3
    input_shape =  [128, 128, None, 1]
    output_shape = [128, 128, None, output_channels]

    train_gen = CustomDataGen(train_patients, cohort = 'train', data_path=data_path).get_gen
    val_gen   = CustomDataGen(val_patients, cohort = 'val', data_path=data_path).get_gen

    output_signature = (tf.TensorSpec(shape=input_shape, dtype=tf.float32), 
                        tf.TensorSpec(shape=output_shape, dtype=tf.float32))

    train_ds = tf.data.Dataset.from_generator(train_gen, output_signature = output_signature)
    val_ds = tf.data.Dataset.from_generator(val_gen, output_signature = output_signature)

    batch_size = 1
    train_ds = train_ds.shuffle(50, seed = 42, reshuffle_each_iteration=True).batch(batch_size).prefetch(-1)
    val_ds = val_ds.batch(batch_size).prefetch(-1)


    if continue_training:

        Path(f'models').mkdir(parents=True, exist_ok=True)
        if not os.path.exists(f'models/{model_name}.h5'):
            run[f'{model_name}'].download(f'models/{model_name}.h5')
        model = tf.keras.models.load_model(f'models/{model_name}.h5', 
                                            compile = False, 
                                            custom_objects = {'ResizeAndConcatenate':ResizeAndConcatenate})    
        print('model loaded')
    else:
        shape = [128, 128, None]
        input_shape = shape + [1]
        output_shape = shape + [output_channels]
        inputs = tf.keras.Input(shape = input_shape)
        tf.keras.backend.clear_session()
        
        unet3 = unet3plus(inputs,
                        filters = [8, 16, 32, 64, 128],
                        rank = 3,  # dimension
                        out_channels = output_channels,
                        add_dropout = 0,
                        dropout_rate = 0.3,
                        kernel_size = 3,
                        encoder_block_depth= 1,
                        decoder_block_depth = 1,
                        pool_size = (2, 2, 2), # This can be either a tuple or int for same pooling across dims
                        skip_type = 'encoder',
                        batch_norm = 1,
                        skip_batch_norm = 1,
                        activation = tf.keras.layers.LeakyReLU(alpha =0.01),#'relu',
                        out_activation = 'softmax',
                        CGM = 0,
                        deep_supervision = 1) # 1 or 0 to add deep_supervision

        model = tf.keras.Model(inputs = inputs, outputs = unet3.outputs())

    model.compile(loss =focal_tversky_loss, optimizer=tf.keras.optimizers.Adam(), loss_weights = [0.25,0.25,0.25,0.25,1])
        
    model.summary()

    es = EarlyStopping(monitor='val_loss', 
                   mode='min', 
                   verbose = 1, 
                   patience = 50)
    
    mc = ModelCheckpoint(f'models/{model_name}.h5',
                        monitor='val_loss',
                        mode='min', 
                        verbose=1)
    
    eval_every_epoch = CustomCallback(model_name = model_name, 
                                    all_patients = all_patients, 
                                    run = run, 
                                    data_path = data_path,
                                    output_signature = output_signature,
                                    save_every = 25)
    
    neptune_callback = NeptuneCallback(run=run) 

    # evaluate(model_name, 
    #          all_patients, 
    #          run, 
    #          data_path, 
    #          output_signature)

    model.fit(train_ds, 
        validation_data = val_ds, 
        epochs=100, 
        callbacks=[es, mc, eval_every_epoch, neptune_callback])
    

    run.stop()

if __name__ == "__main__":
    main()