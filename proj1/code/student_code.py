import numpy as np

def my_imfilter(image, filter):
  """
  Apply a filter to an image. Return the filtered image.

  Args
  - image: numpy nd-array of dim (m, n, c)
  - filter: numpy nd-array of dim (k, k)
  Returns
  - filtered_image: numpy nd-array of dim (m, n, c)

  HINTS:
  - You may not use any libraries that do the work for you. Using numpy to work
   with matrices is fine and encouraged. Using opencv or similar to do the
   filtering for you is not allowed.
  - I encourage you to try implementing this naively first, just be aware that
   it may take an absurdly long time to run. You will need to get a function
   that takes a reasonable amount of time to run so that the TAs can verify
   your code works.
  - Remember these are RGB images, accounting for the final image dimension.
  """

  assert filter.shape[0] % 2 == 1
  assert filter.shape[1] % 2 == 1
  #print(image.shape)
  #print(filter.shape)
  # create a bigger image with padding
  padded_x_size = image.shape[0] + ((filter.shape[0] - 1) * 2)
  padded_y_size = image.shape[1] + ((filter.shape[1] - 1) * 2)
  padding = np.zeros((padded_x_size, padded_y_size, image.shape[2]))
  #print(padding.shape)
  img_x_min = filter.shape[0] - 1
  img_x_max = padding.shape[0] - filter.shape[0] + 1
  img_y_min = filter.shape[1] - 1
  img_y_max = padding.shape[1] - filter.shape[1] + 1 # outer boundary is not inclusive
  #print("hello")
  padding[img_x_min:img_x_max, img_y_min: img_y_max] = image


  filtered_image = np.zeros(image.shape)
  for x in range(image.shape[0]):
    for y in range(image.shape[1]):
      for c in range(image.shape[2]):  # every colorchannel
         sub_img = padding[x:filter.shape[0] + x, y:filter.shape[1] + y, c]
         #print(sub_img.shape)
         filtered_image[x][y][c] = np.sum(np.multiply(sub_img, filter))


  ############################
  ### TODO: YOUR CODE HERE ###

  #raise NotImplementedError('`my_imfilter` function in `student_code.py` ' +
  #  'needs to be implemented')

  ### END OF STUDENT CODE ####
  ############################

  return filtered_image

def create_hybrid_image(image1, image2, filter):
  """
  Takes two images and creates a hybrid image. Returns the low
  frequency content of image1, the high frequency content of
  image 2, and the hybrid image.

  Args
  - image1: numpy nd-array of dim (m, n, c)
  - image2: numpy nd-array of dim (m, n, c)
  Returns
  - low_frequencies: numpy nd-array of dim (m, n, c)
  - high_frequencies: numpy nd-array of dim (m, n, c)
  - hybrid_image: numpy nd-array of dim (m, n, c)

  HINTS:
  - You will use your my_imfilter function in this function.
  - You can get just the high frequency content of an image by removing its low
    frequency content. Think about how to do this in mathematical terms.
  - Don't forget to make sure the pixel values are >= 0 and <= 1. This is known
    as 'clipping'.
  - If you want to use images with different dimensions, you should resize them
    in the notebook code.
  """

  assert image1.shape[0] == image2.shape[0]
  assert image1.shape[1] == image2.shape[1]
  assert image1.shape[2] == image2.shape[2]

  ############################
  ### TODO: YOUR CODE HERE ###
  low_frequencies = my_imfilter(image1, filter)
  img2_low_freq = my_imfilter(image2, filter)
  print("cat min max: ", np.min(img2_low_freq), np.max(img2_low_freq))
  print("dog min max:", np.min(low_frequencies), np.max(low_frequencies))
  print("cat original min max", np.min(image2), np.max(image2))

  high_frequencies = image2 - img2_low_freq
  high_frequencies = (high_frequencies) / 2
  print("high freq min max", np.min(high_frequencies), np.max(high_frequencies))
  hybrid_image = (high_frequencies + low_frequencies)


  #raise NotImplementedError('`create_hybrid_image` function in ' +
   # '`student_code.py` needs to be implemented')

  ### END OF STUDENT CODE ####
  ############################

  return low_frequencies, high_frequencies, hybrid_image
