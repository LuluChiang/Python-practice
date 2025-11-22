from io import IncrementalNewlineDecoder
from typing import List, Optional

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Tree:
    def __init__(self):
        self.root = None

class Solution:
# 94. Binary Tree Inorder Traversal
# Given the root of a binary tree, return the inorder traversal of its nodes' values.
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        if root == None:
            return []         
        opt = [root.val]
        return self.inorderTraversal(root.left) + opt + self.inorderTraversal(root.right)

# 100. Same Tree
# Given the roots of two binary trees p and q, write a function to check if they are the same or not.
# Two binary trees are considered the same if they are structurally identical, and the nodes have the same value.
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if p == None or q == None:
            return p == q
        
        return p.val == q.val and self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)

# 101. Symmetric Tree
# Given the root of a binary tree, check whether it is a mirror of itself (i.e., symmetric around its center).
# Input: root = [1,2,2,3,4,4,3]
# Output: true
#
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        if not root:
            return True
        else:  
            if root.left == None or root.right == None:
                return root.left == root.right


            return self.isMirrorTree(root.left, root.right)

    def isMirrorTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if p == None or q == None:
            return p == q
        
        return p.val == q.val and self.isMirrorTree(p.right, q.left) and self.isMirrorTree(p.left, q.right)

# 96. Unique Binary Search Trees
# Given an integer n, return the number of structurally unique BST's (binary search trees) which has exactly n nodes of unique values from 1 to n.
#
    def numTrees(self, n: int) -> int:
        if n == 0:
            return 1
        else:
            sum = 0
            for idx in range(1 ,n):
                leftn = idx - 1
                rightn = n - idx
                sum += self.numTrees(leftn) + self.numTrees(rightn)
        return sum

# 104. Maximum Depth of Binary Tree
# Given the root of a binary tree, return its maximum depth.
# A binary tree's maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.
#
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if root == None:
            return 0
        if root.left == None and root.right == None:
            return 1
        else:
            return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))
        
# 110. Balanced Binary Tree
# Given a binary tree, determine if it is height-balanced.
# For this problem, a height-balanced binary tree is defined as:
# a binary tree in which the left and right subtrees of every node differ in height by no more than 1.
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        if root == None:
            return True
        
        return abs(self.maxDepth(root.left) - self.maxDepth(root.right)) < 2 and self.isBalanced(root.left) and self.isBalanced(root.right)

# 111. Minimum Depth of Binary Tree
# Given a binary tree, find its minimum depth.
# The minimum depth is the number of nodes along the shortest path from the root node down to the nearest leaf node.
# Note: A leaf is a node with no children.
    def minDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        if None in [root.left, root.right]:
            return max(self.minDepth(root.left), self.minDepth(root.right)) + 1
        else:
            return min(self.minDepth(root.left), self.minDepth(root.right)) + 1

# 112. Path Sum
# Given the root of a binary tree and an integer targetSum, return true if the tree has a root-to-leaf path such that adding up all the values along the path equals targetSum.
# A leaf is a node with no children.
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        if not root:
            return False
        elif not root.left and not root.right:
            return root.val == targetSum
        else:
            targetchildsum = targetSum - root.val
            return self.hasPathSum(root.left, targetchildsum) or self.hasPathSum(root.right, targetchildsum)


# 226. Invert Binary Tree
# Given the root of a binary tree, invert the tree, and return its root.
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
# 2025/11/19    
    def invertTree2(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root: return None
        self.invertTree2(root.left)
        self.invertTree2(root.right)
        root.left, root.right = root.right, root.left
        return root

# 2024/9/12    
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        
        # if root == None:
        #     return None
        # or use:
        if not root:  #in python = if root != None
            return
        
        # temp = root.left
        # root.left = root.right
        # root.right = temp
        # or use:
        (root.left, root.right) = (root.right, root.left) #tuple assignment

        # recursive part
        self.invertTree(root.left)
        self.invertTree(root.right)
    
        return root

# Binary Tree Traversal
# 2025/11/22
#   def DFS_Left_preorder(self, root):
#   def DFS_Left_inorder(self, root):
#   def DFS_Left_postorder(self, root):
#   def DFS_Right_preorder(self, root):
#   def DFS_Right_inorder(self, root):
#   def DFS_Right_postorder(self, root):
    def DFS_Left_preorder(self, root)-> list: 
        if not root: 
            return None
        #print(root.val)
        rtn_list = [root.val]
        #rtn_list += self.DFS_Left_preorder(root.left)
        rtn_list.extend(self.DFS_Left_preorder(root.left))
        #rtn_list += self.DFS_Left_preorder(root.right)
        rtn_list.extend(self.DFS_Left_preorder(root.right))
        return rtn_list
    def DFS_Left_preorder_iteration(self, root) -> list:
        if not root: 
            return None
        rtn_list = []
        stack = [root]
        while stack:
            node = stack.pop()
            rtn_list.append(node.val)
            #由於stack是LIFO，所以right先放，left後放先出
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)

        return rtn_list
    def DFS_Left_inorder_iteration(self, root) -> list:
        if not root:
            return None
        rtn_list = []
        stack = []
        node = root
        while stack or node:
            #先往左邊走，並且先把自己存起來
            while node:
                stack.append(node)
                node = node.left
            node = stack.pop()
            rtn_list.append(node.val)
            node = node.right
        return rtn_list
    
    def DFS_Left_postorder_iteration(self, root) -> list:
        if not root:
            return None
        rtn_list = []
        stack = [root]
        
        while stack:
            node = stack.pop()
            rtn_list.append(root.val)
            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)

        return rtn_list[::-1]
    
 