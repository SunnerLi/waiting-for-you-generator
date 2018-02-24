from torchvision import transforms
import torchvision_sunner.transforms as sunnertransforms
import torchvision_sunner.data as sunnerData
import numpy as np
import cv2

if __name__ == '__main__':
    dataset = sunnerData.ImageDataset(
        root_list = ['./waiting_for_you_dataset/real_world', './waiting_for_you_dataset/wait'],
        sample_method = sunnerData.OVER_SAMPLING,
        use_cv = False,
        transform = transforms.Compose([
            sunnertransforms.Rescale((160, 320), use_cv = False),
            sunnertransforms.ToTensor(),
            sunnertransforms.ToFloat(),
            sunnertransforms.Transpose(sunnertransforms.BHWC2BCHW),
            sunnertransforms.Normalize()
        ]) 
    )
    loader = sunnerData.ImageLoader(dataset, batch_size=4, shuffle=True, num_workers = 2)

    for img, _ in loader:
        img_np = img.numpy()
        img = sunnertransforms.tensor2Numpy(img, transform = transforms.Compose([
            sunnertransforms.UnNormalize(),
            sunnertransforms.Transpose(sunnertransforms.BCHW2BHWC),
        ])) 
        print('mean: ', np.mean(img), '\tstd: ', np.std(img), '\tmin: ', np.min(img), '\tmax: ', np.max(img))

        # Show
        img = img[0]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = img.astype(np.uint8)
        cv2.imshow('show', img)
        cv2.waitKey()

        # Done
        break