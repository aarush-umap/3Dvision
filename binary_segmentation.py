import numpy as np
from collections import deque


def region_growing_alg(image):
    """Labels each part of image pixel by pixel
    
    Algorithm lables one connected component at a time
    """
    labels = set()
    labeled = np.zeros_like(image)
    visited = set()
    n, m = image.shape

    def dfs(i, j, label):
        """ Labels all valid neighbors"""
        
        labeled[i, j] = label
        visited.add((i, j))

        for x, y in [(i + 1 , j), (i , j + 1), (i - 1 , j), (i , j - 1)]:
            if (0 <= x < n and 0 <= y < m and (x, y) not in visited 
                and image[x, y] == 1):
                dfs(x, y, label)

    label = 0
    for i in range(n):
        for j in range(m):
            if (i, j) not in visited and image[i, j] == 1:
                label += 1
                labels.add(label)
                dfs(i, j, label)

    return labels, labeled

def region_grow_bfs(image): 
    """Labels each part of image pixel by pixel
    
    Algorithm lables one connected component at a time iteratively
    """
    labels = set()
    labeled = np.zeros_like(image)
    visited = set()
    n, m = image.shape

    def bfs(i, j, label):
        """ Labels all valid neighbors"""
        
        queue = deque()
        queue.append((i, j))

        while(queue):
            curr_i, curr_j = queue.popleft()
            labeled[curr_i, curr_j] = label
            visited.add((curr_i, curr_j)) # redundant but secure

            for x, y in [(curr_i + 1 , curr_j), (curr_i , curr_j + 1), (curr_i - 1 , curr_j), (curr_i , curr_j - 1)]:
                if (0 <= x < n and 0 <= y < m and (x, y) not in visited and image[x, y] == 1):
                    visited.add((x, y)) # can't forget to add stuff before it goes into the queue!
                    queue.append((x, y))


    label = 0
    for i in range(n):
        for j in range(m):
            if (i, j) not in visited and image[i, j] == 1:
                label += 1
                labels.add(label)
                bfs(i, j, label)

    return labels, labeled

def seq_label_alg(image):
    """ Labels each part of the image row by row

    Algorithm labels each pixel based on the labeling of it's neighbors (defined by 6-C)    
    """
    labeled = np.zeros_like(image)
    equiv_table = {0:0}
    label = 1
    n, m = image.shape

    def valid_pos(i, j):
        """Checks if indices are in bounds"""
        return 0<=i<n and 0<=j<m
    
    def find(a):
        """Finds parent of current label"""
        if equiv_table[a] != a:
            equiv_table[a] = find(equiv_table[a])
        return equiv_table[a]
    
    def union(a, b):
        """Merges two labels in equivalence table
        
        Assigns smallest label value as true parent
        """
        a_parent = find(a)
        b_parent = find(b)
        if a_parent == b_parent:
            return
        elif a_parent < b_parent:
            equiv_table[b_parent] = a_parent
            return
        else:
            equiv_table[a_parent] = b_parent

        
    for i in range(n):
        for j in range(m):
            if image[i,j] != 0:
                # check pos D
                if valid_pos(i-1,j-1) and labeled[i-1, j-1]:
                    labeled[i, j] = labeled[i-1, j-1]

                # checks both pos B and pos C 
                elif (valid_pos(i-1, j) and labeled[i-1, j] and
                      valid_pos(i, j - 1) and labeled[i, j - 1]):
                    labeled[i, j] = min(labeled[i-1, j], labeled[i, j-1])
                    union(labeled[i-1, j], labeled[i, j-1])

                # checks only pos B
                elif valid_pos(i-1, j) and labeled[i-1, j]:
                    labeled[i, j] = labeled[i-1, j]

                # checks only pos C
                elif valid_pos(i, j - 1) and labeled[i, j - 1]:
                    labeled[i, j] = labeled[i, j - 1]
                
                # creates new label
                else:
                    labeled[i, j] = label
                    equiv_table[label] = label
                    label += 1

    # second pass to finalize labels
    labels = set([0])
    for i in range(n):
        for j in range(m):
            labeled[i, j] = find(labeled[i, j])
            labels.add(labeled[i, j])

    return labels, labeled

def skeletonization(image):
    '''Thins image using Zhang Suen's thinning algorithm
    
    Parameters
    * image (ndarray) - binary matrix with shape (H,W)

    Returns
    * frames (list) - a collection of iterations in thinning process
    '''

    frames = []
    H, W = image.shape
    image = np.pad(image, (1,1), mode='constant', constant_values=(0,0))

    def get_neighbors(i, j):
        return np.array([
            image[i - 1, j], # P2
            image[i - 1, j + 1], # P3
            image[i, j + 1], # P4
            image[i + 1, j + 1], # P5
            image[i + 1, j], # P6
            image[i + 1, j - 1], # P7
            image[i, j - 1], # P8
            image[i - 1, j - 1]  # P9
        ])
    
    def get_transitions(neighbors):
        n = neighbors + neighbors[0]
        count = 0
        for n1, n2 in zip(n, n[1:]):
            if n1 == 0 and n2 == 1:
                count += 1

        return count

    changed_first = changed_second = [('flag', 'flag')]
    while changed_first or changed_second:
        changed_first = []
        for i in range(1, H):
            for j in range(1, W):
                if image[i,j] == 1:
                    neighbors = get_neighbors(i, j)
                    if (2 <= len(neighbors[neighbors == 1]) <= 6 and
                        get_transitions(neighbors) == 1 and
                        neighbors[0] * neighbors[2] * neighbors[4] == 0 and
                        neighbors[2] * neighbors[4] * neighbors[6] == 0 ):
                        changed_first.append((i, j))
        for i, j in changed_first:
            image[i, j] = 0


        changed_second = []
        for i in range(1, H):
            for j in range(1, W):
                if image[i,j] == 1:
                    neighbors = get_neighbors(i, j)
                    if (2 <= len(neighbors[neighbors == 1]) <= 6 and
                        get_transitions(neighbors) == 1 and
                        neighbors[0] * neighbors[2] * neighbors[6] == 0 and
                        neighbors[0] * neighbors[4] * neighbors[6] == 0 ):
                        changed_second.append((i, j))
        for i, j in changed_second:
            image[i, j] = 0       

    return image[1:H, 1:W] 

def color_segmentations(labels, labeled_image):
    """Returns new image with unique colors for each object
    
    Parameters
    * labels (set) - a set of all unique labels
    * labeled_image (ndarrray) - matrix with shape (H, W) where pos[i, j] = label
    """
    n,m = labeled_image.shape
    colors = {l: tuple(np.random.rand(3)) for l in labels}
    colored = np.zeros(shape=(n, m, 3))

    colors[0] = (0, 0, 0)
    for l, c in colors.items():
        colored[labeled_image == l] = c

    return colored