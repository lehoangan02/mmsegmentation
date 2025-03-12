from mmseg.apis import inference_model, init_model, show_result_pyplot
import mmcv

config_file = 'pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py'
checkpoint_file = 'pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

# build the model from a config file and a checkpoint file
model = init_model(config_file, checkpoint_file, device='cpu')

# test a single image and show the results
img = 'demo/Amsterdam.jpg'  # or img = mmcv.imread(img), which will only load it once
result = inference_model(model, img)
result_name = 'result.png'
show_result_pyplot(
        model,
        img,
        result,
        title=result_name,
        opacity=0.5,
        with_labels=False,
        draw_gt=True,
        show=False if result_name is not None else True,
        out_file=result_name)
print(type(result))