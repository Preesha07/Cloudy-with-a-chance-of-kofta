def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'pix2pix_attn_nir_v4c':
        from .pix2pix_attn_nir_v4c_model import Pix2Pix_attn_NIR_v4c_Model
        model = Pix2Pix_attn_NIR_v4c_Model()

    elif opt.model == 'pix2pix_attn_nir_v4c_ffl_sar':
        from .pix2pix_attn_nir_v4c_ffl_sar_model import Pix2Pix_attn_NIR_v4c_FFL_SAR_Model
        model = Pix2Pix_attn_NIR_v4c_FFL_SAR_Model()

    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
