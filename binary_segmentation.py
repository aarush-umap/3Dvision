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
    labeled = np.zeros_like(image)
    equiv_table = {0:0}
    label = 1
    n, m = image.shape

    def valid_pos(i, j):
        return 0<=i<n and 0<=j<m
    
    def find(a):
        if equiv_table[a] != a:
            equiv_table[a] = find(equiv_table[a])
        return equiv_table[a]
    
    def union(a, b):
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

                elif (valid_pos(i-1, j) and labeled[i-1, j] and
                      valid_pos(i, j - 1) and labeled[i, j - 1]):
                    labeled[i, j] = min(labeled[i-1, j], labeled[i, j-1])
                    union(labeled[i-1, j], labeled[i, j-1])

                elif valid_pos(i-1, j) and labeled[i-1, j]:
                    labeled[i, j] = labeled[i-1, j]

                elif valid_pos(i, j - 1) and labeled[i, j - 1]:
                    labeled[i, j] = labeled[i, j - 1]
                
                else:
                    labeled[i, j] = label
                    equiv_table[label] = label
                    label += 1

    labels = set([0])
    for i in range(n):
        for j in range(m):
            labeled[i, j] = find(labeled[i, j])
            labels.add(labeled[i, j])

    return labels, labeled

def color_segmentations(labels, labeled_image):
    n,m = labeled_image.shape
    colors = {l: tuple(np.random.rand(3)) for l in labels}
    colored = np.zeros(shape=(n, m, 3))

    colors[0] = (0, 0, 0)
    for l, c in colors.items():
        colored[labeled_image == l] = c

    return colored