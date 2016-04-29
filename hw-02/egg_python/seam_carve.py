from numpy import zeros

def seam_carve(img, mode, mask=None):
    # dummy implementation — delete rightmost column of image
    resized_img = img[:, :-1, :]
    if mask is None:
        resized_mask = None
    else:
        resized_mask = mask[:, :-1]

    carve_mask = zeros(img.shape[0:2])
    carve_mask[:, -1] = 1
    return (resized_img, resized_mask, carve_mask)

