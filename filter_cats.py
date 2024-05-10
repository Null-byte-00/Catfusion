import pygame
import os
import shutil

pygame.init()
screen = pygame.display.set_mode((500,500))

if not os.path.exists('cat_images'):
    os.makedirs('cat_images')

image_index = 1872

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                image_index -= 1
            if event.key == pygame.K_RIGHT:
                image_index += 1
            if event.key == pygame.K_UP:
                print('Cat image moved to cat_images')
                shutil.copyfile('cats/' + str(image_index) + '.jpg', 'cat_images/' + str(image_index) + '.jpg')
                num_files = len([name for name in os.listdir('cat_images')])
                print('Number of cat images in cat_images:', num_files)

    try:
        image = pygame.image.load('cats/' + str(image_index) + '.jpg')
        image = pygame.transform.scale(image, (500,500))
        screen.blit(image, (0,0))
    except:
        pass

    pygame.display.update()