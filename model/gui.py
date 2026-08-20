"""
Interactive Galaxy Explorer GUI

This interactive visualization tool allows you to browse through the validation set and examine the 
model's predictions in real-time. The GUI displays each galaxy image using the Magma colormap 
(commonly used in astronomy visualization) alongside the model's softmax probability distribution
across all four classes.

Controls
--------
- LEFT/RIGHT Arrow Keys: Navigate through validation samples
- R Key: Jump to a random sample
- M Key: Toggle Magma colormap on/off
- Q Key: Quit the application

The visualization highlights the predicted class with a green bar, making it easy to spot correct 
classifications and identify failure cases where the model might confuse similar morphologies 
(e.g., smooth round vs. smooth cigar galaxies).
"""

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import pygame
import numpy as np
import torch
import matplotlib.pyplot as plt

class GalaxyExplorerGUI:
    """
    Interactive GUI for exploring galaxy classifications.
    
    Displays galaxy images alongside model predictions with real-time navigation.
    Supports toggling between standard RGB and Magma colormap visualization.
    
    Parameters
    ----------
    model : ModelInterface
        Trained model for galaxy classification.
    x_val : torch.Tensor
        Validation images tensor of shape (N, C, H, W).
    y_val : torch.Tensor
        One-hot encoded validation labels of shape (N, num_classes).
    device : torch.device
        Device for running inference (CPU or CUDA).
    
    Attributes
    ----------
    current_idx : int
        Index of currently displayed sample.
    num_samples : int
        Total number of validation samples.
    predictions : np.ndarray
        Current model prediction probabilities.
    use_magma : bool
        Whether to apply Magma colormap to displayed image.
    """
    def __init__(self, model, x_val, y_val, device):
        self.model = model
        self.x_val = x_val  
        self.y_val = y_val  
        self.device = device
        
        self.current_idx = 0
        self.num_samples = len(x_val)
        self.predictions = np.zeros(4)
        
        # New toggle state
        self.use_magma = False
        
        pygame.init()
        self.WIDTH, self.HEIGHT = 1000, 600
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("S4 GALAXY EXPLORER")
        
        self.CANVAS_SIZE = 448
        self.class_labels =  ["Smooth Round", "Smooth Cigar", "Edge-on Disk", "Unbarred Spiral"] # Class names for GalaxyMNIST
        
        self.COLOR_BG = (5, 5, 8)
        self.COLOR_ACCENT = (255, 160, 60) 
        self.COLOR_SUCCESS = (0, 255, 120)
        self.COLOR_FAILURE = (255, 50, 50)
        
        self.font = pygame.font.SysFont("monospace", 15)
        self.big_font = pygame.font.SysFont("monospace", 22, bold=True)
        
        self.cmap = plt.get_cmap('magma')
        self.update_sample(0)

    def update_sample(self, delta):
        """
        Update the currently displayed sample and compute predictions.
        
        Parameters
        ----------
        delta : int
            Offset to add to current index (wraps around at boundaries).
        """
        self.current_idx = (self.current_idx + delta) % self.num_samples
        self.model.eval()
        with torch.no_grad():
            img_tensor = self.x_val[self.current_idx].unsqueeze(0).to(self.device)
            probs = self.model(img_tensor)
            self.predictions = probs.squeeze().cpu().numpy()

    def draw(self):
        """
        Render the current frame of the GUI.
        
        Displays the galaxy image (with optional Magma colormap), prediction bars,
        sample metadata, and keyboard controls. Highlights correct predictions in
        green and incorrect predictions in red.
        """
        self.screen.fill(self.COLOR_BG)
        
        # x_val is always (N, C, H, W) -- so raw_img here is always channel-first,
        # never (H, W, C). The "or [H, W, C]" in the old comment was never actually
        # reachable, and assuming it was is what broke grayscale rendering below.
        raw_img = self.x_val[self.current_idx].numpy()
        
        if self.use_magma:
            # mean over the channel axis works whether C=1 (trivial mean, same as
            # squeezing) or C=3 (proper RGB->gray average) -- no need to special-case
            gray_img = raw_img.mean(axis=0)
            
            magma_img = self.cmap(gray_img) 
            rgb_render = (magma_img[:, :, :3] * 255).astype(np.uint8)
        else:
            # Standard RGB Render
            # Pygame wants (H, W, 3) uint8. Grayscale data is (1, H, W) -- transposing
            # alone gives (H, W, 1), which pygame's make_surface rejects (last dim must
            # be exactly 3), so we tile the single channel across R/G/B instead.
            rgb_render = raw_img.transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
            if rgb_render.shape[-1] == 1:
                rgb_render = np.repeat(rgb_render, 3, axis=-1)
            
            # Ensure uint8 [0, 255]
            if rgb_render.max() <= 1.0:
                rgb_render = (rgb_render * 255).astype(np.uint8)

        # 2. Create Pygame surface (Transpose from Row-Major to Width-Major)
        surface = pygame.surfarray.make_surface(rgb_render.transpose(1, 0, 2))
        scaled_img = pygame.transform.scale(surface, (self.CANVAS_SIZE, self.CANVAS_SIZE))
        self.screen.blit(scaled_img, (40, 60))

        pygame.draw.rect(self.screen, self.COLOR_ACCENT, (40, 60, self.CANVAS_SIZE, self.CANVAS_SIZE), 2)
        
        true_label_idx = torch.argmax(self.y_val[self.current_idx]).item()
        meta_txt = self.font.render(f"Sample: {self.current_idx} | Truth: {self.class_labels[true_label_idx]} | Magma: {'ON' if self.use_magma else 'OFF'}", True, (200, 200, 200))
        self.screen.blit(meta_txt, (40, 35))

        x_off = 520
        top_pred = np.argmax(self.predictions)
        self.screen.blit(self.big_font.render("MODEL PREDICTION", True, self.COLOR_ACCENT), (x_off, 60))
        
        for i, label in enumerate(self.class_labels):
            prob = self.predictions[i]
            bar_y = 120 + (i * 60)

            if i == top_pred and i == true_label_idx:
                color = self.COLOR_SUCCESS
            elif i == top_pred and i != true_label_idx:
                color = self.COLOR_FAILURE
            else:
                color = (150, 150, 150)

            txt = self.font.render(f"{label}: {prob*100:4.1f}%", True, color)
            self.screen.blit(txt, (x_off, bar_y))
            
            pygame.draw.rect(self.screen, (20, 20, 30), (x_off, bar_y + 25, 400, 15))
            pygame.draw.rect(self.screen, color, (x_off, bar_y + 25, int(prob * 400), 15))

        footer = self.font.render("[L/R] Change | [R] Rand | [M] Magma | [Q] Quit", True, self.COLOR_ACCENT)
        self.screen.blit(footer, (40, 530))

    def run(self):
        """
        Main event loop for the GUI application.
        
        Handles keyboard input for navigation (arrow keys, R for random),
        colormap toggling (M key), and quitting (Q key). Runs until the
        user closes the window or presses Q.
        """
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RIGHT: self.update_sample(1)
                    if event.key == pygame.K_LEFT: self.update_sample(-1)
                    if event.key == pygame.K_r: self.update_sample(np.random.randint(0, self.num_samples))
                    if event.key == pygame.K_m: self.use_magma = not self.use_magma # Toggle
                    if event.key == pygame.K_q: running = False
            self.draw()
            pygame.display.flip()
        pygame.quit()

