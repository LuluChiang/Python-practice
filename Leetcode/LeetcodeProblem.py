from typing import List


class Solution:
# 1. Two Sum
# Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
# You may assume that each input would have exactly one solution, and you may not use the same element twice.
# You can return the answer in any order.
#  
# Date: 2021/09/23
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        if(len(nums)) < 2:
            return None
        for i in range(len(nums)):
            for j in range(i+1,len(nums)):
                if nums[i] + nums[j] == target:
                    print(str(i) + " + " +  str(j) + " = " + str(target))
                    list_ans = [i, j]
                    return list_ans

# 66. Plus One
# You are given a large integer represented as an integer array digits, where each digits[i] is the ith digit of the integer. 
# The digits are ordered from most significant to least significant in left-to-right order. 
# The large integer does not contain any leading 0's.
# Increment the large integer by one and return the resulting array of digits.
#
# Date: 2021/09/23
    def plusOne(self, digits: List[int]) -> List[int]:        
        if digits[-1] % 10 != 9:
            digits[-1] += 1
            return digits
        else:
            digits[-1] = 0
            if(len(digits) == 1):
                digits.insert(0, 1)
            else:                
                digits = self.plusOne(digits[:-1])
                digits.insert(len(digits), 0)
            return digits
# Date: 2021/09/24 
    def plusOne2(self, digits: List[int]) -> List[int]:        
        for i in range(len(digits)-1 , -1 , -1):
            if digits[i] != 9:
                digits[i] += 1 
                return digits
            else:
                digits[i] = 0
                if i == 0:
                    digits.insert(0, 1)
                    return digits
                else:
                    continue

# 3. Longest Substring Without Repeating Characters
# Given a string s, find the length of the longest substring without repeating characters.
# Input: s = "abcabcbb"
# Output: 3
# Explanation: The answer is "abc", with the length of 3.
    # answer in intuition
    def lengthOfLongestSubstring(self, s: str) -> int:
        if s == "":
            return 0
        max_count = 1
        for base in range(0, len(s)):
            tmp_str = s[base]
            count = 1           

            for idx in range(base + 1, len(s)):
                if(s[idx] not in tmp_str):
                    tmp_str += s[idx]
                    count += 1
                    max_count = max(max_count, count)
                else:
                    break         
        return max_count
    # O(n) solution  
    # the key conception: when encounter the same character 'x' already apeared in tmp_str,
    #   we reassign the tmp_str from the x's index + 1, and add 'x' in the tail 
    def lengthOfLongestSubstring2(self, s):
        tmp_str = ""
        max_count = 0
        for idx in range(0, len(s)):
            if s[idx] not in tmp_str:
                tmp_str += s[idx]
            else:
                idx_samechar = tmp_str.index(s[idx])
                tmp_str = tmp_str[idx_samechar + 1:] + s[idx]
            max_count = max(max_count, len(tmp_str))
        return max_count

# 9. Palindrome Number
# Given an integer x, return true if x is palindrome integer.
# An integer is a palindrome when it reads the same backward as forward. For example, 121 is palindrome while 123 is not.
#   -121: From left to right, it reads -121. From right to left, it becomes 121-. Therefore it is not a palindrome.
#
    def isPalindrome(self, x: int) -> bool:
        if str(x) == str(x)[::-1]:
            return True
        else:
            return False

# 5. Longest Palindromic Substring
# Given a string s, return the longest palindromic substring in s.
#
#   Note: This solution exceed limit time after submit to leetcode online, check this later.
# 10/14 update: 
#   TODO: After checking others discusstion, this probelm can be solved in O(N)
#
    def IsStrPalindrome(self, s:str) -> bool:
        if s == s[::-1]:
            return True
        else:
            return False

    def longestPalindrome(self, s: str) -> str:
        tmp_str = max_str = ""
        
        for idx in range(len(s)):
            for idx2 in range(idx, len(s)):
                tmp_str = s[idx:idx2 + 1]
                if self.IsStrPalindrome(tmp_str) and len(max_str) < len(tmp_str):
                    max_str = tmp_str
        return max_str

# 14. Longest Common Prefix
# Write a function to find the longest common prefix string amongst an array of strings.
# If there is no common prefix, return an empty string "".
#
    def longestCommonPrefix(self, strs: List[str]) -> str:
        pre_str = ""
        for idx in range(1, len(strs[0]) + 1):
            tmp_str = strs[0][:idx]
            for num in range(1, len(strs)):
                if tmp_str != strs[num][:idx]:
                    return pre_str
            pre_str = tmp_str
        return pre_str

# 13. Roman to Integer
# Roman numerals are represented by seven different symbols: I, V, X, L, C, D and M.
# Symbol       Value
# I             1
# V             5
# X             10
# L             50
# C             100
# D             500
# M             1000
#For example, 2 is written as II in Roman numeral, just two one's added together. 12 is written as XII, which is simply X + II. The number 27 is written as XXVII, which is XX + V + II.
# Roman numerals are usually written largest to smallest from left to right. However, the numeral for four is not IIII. Instead, the number four is written as IV. Because the one is before the five we subtract it making four. The same principle applies to the number nine, which is written as IX. There are six instances where subtraction is used:
# I can be placed before V (5) and X (10) to make 4 and 9. 
# X can be placed before L (50) and C (100) to make 40 and 90. 
# C can be placed before D (500) and M (1000) to make 400 and 900.
# Given a roman numeral, convert it to an integer.
#
    def romanToInt(self, s: str) -> int:
        if s == "":
            return 0
        dicRom = {
            'I': 1,
            'V': 5,
            'X': 10,
            'L': 50,
            'C': 100,
            'D': 500,
            'M': 1000,
        }
        total = 0
        
        for idx in range(len(s) - 1):                
            if dicRom[s[idx]] >= dicRom[s[idx + 1]]:
                total += dicRom[s[idx]]
            else: 
                total -= dicRom[s[idx]]
        return total + dicRom[s[len(s) - 1]]

# 20. Valid Parentheses
# Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.
# An input string is valid if:
# Open brackets must be closed by the same type of brackets.
# Open brackets must be closed in the correct order.
#
    def isValidParentheses(self, s: str) -> bool:
        list_left = ['(', '[', '{'] 
        list_right = [')', ']', '}'] 
        idx = 0
        tmp_list = list(s)
        while idx < (len(tmp_list)):
            if tmp_list[idx] in list_right:
                if idx == 0:
                    return False
                if tmp_list[idx - 1] == list_left[list_right.index(tmp_list[idx])]:
                    tmp_list.pop(idx)
                    tmp_list.pop(idx - 1)
                    idx -=1
                    continue
            idx += 1
        return tmp_list == []

    # 10/12 update: try using dictionary
    def isValidParentheses2(self, s: str) -> bool:
        dic_paren = { ')':'(', ']':'[',  '}':'{' }
        list_stack = []

        for each_paren in s:
            if each_paren in dic_paren.values():
                list_stack.append(each_paren)
            elif each_paren in dic_paren:
                if list_stack == []:
                    return False
                if dic_paren[each_paren] != list_stack.pop():
                    return False
        return list_stack == []

# 26. Remove Duplicates from Sorted Array
# Given an integer array nums sorted in non-decreasing order, remove the duplicates in-place such that each unique element appears only once. The relative order of the elements should be kept the same.
# Since it is impossible to change the length of the array in some languages, you must instead have the result be placed in the first part of the array nums. More formally, if there are k elements after removing the duplicates, then the first k elements of nums should hold the final result. It does not matter what you leave beyond the first k elements.
# Return k after placing the final result in the first k slots of nums.
# Do not allocate extra space for another array. You must do this by modifying the input array in-place with O(1) extra memory.
# Custom Judge:
# The judge will test your solution with the following code:
#
    def removeDuplicates(self, nums: List[int]) -> int:
        tmp_list = []
        count = 0
        for num in nums:
            if num not in tmp_list:
                count += 1
                tmp_list.append(num)
        
        return count

#20240804
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        slow = 0
        
        for fast in range(1, len(nums)):
            if nums[slow] == nums[fast]: 
                fast += 1
            else:
                slow += 1 
                nums[slow] = nums[fast]
        return slow+1
            
        return count

# 53. Maximum Subarray - easy
# Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.
# A subarray is a contiguous part of an array.
# Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
# Output: 6
# Explanation: [4,-1,2,1] has the largest sum = 6.
# Note: Using Dynamic Programing!!
#  
    def maxSubArray(self, nums: List[int]) -> int:
        maxsum_for_curr = nums[0]
        idx = 1
        maxsum = nums[0]
        while idx < len(nums):
            if maxsum_for_curr > 0:
               maxsum_for_curr = maxsum_for_curr + nums[idx]
            else:
               maxsum_for_curr = nums[idx]

            maxsum = max(maxsum, maxsum_for_curr)
            idx += 1
        return maxsum

# 58. Length of Last Word
# Given a string s consisting of some words separated by some number of spaces, return the length of the last word in the string.
# A word is a maximal substring consisting of non-space characters only.
# Input: s = "   fly me   to   the moon  "
# Output: 4
# Explanation: The last word is "moon" with length 4.
#
    def lengthOfLastWord(self, s: str) -> int:
        currword = 0
        lastword = 0
        for char in s:
            if char == ' ':      
                currword = 0
            else:
                currword += 1
            if currword != 0:
                lastword = currword
        return lastword


# 67. Add Binary
# Given two binary strings a and b, return their sum as a binary string.
# Example 1:
# Input: a = "11", b = "1"
# Output: "100"
#
    def addBinary(self, a: str, b: str) -> str:
        list_a = list(a)
        list_b = list(b)
        carry = 0
        sum = ''
        while list_a or list_b or carry:
            if list_a:
                carry += int(list_a.pop())
            if list_b:
                carry += int(list_b.pop())
            
            sum = str(carry % 2) + sum
            carry = carry // 2
        
        return sum

# 69. Sqrt(x)
# Given a non-negative integer x, compute and return the square root of x.
# Since the return type is an integer, the decimal digits are truncated, and only the integer part of the result is returned.
# Note: You are not allowed to use any built-in exponent function or operator, such as pow(x, 0.5) or x ** 0.5.
# Example :
#     Input: x = 8
#     Output: 2
#     Explanation: The square root of 8 is 2.82842..., and since the decimal part is truncated, 2 is returned.
#
    def mySqrt(self, x: int) -> int:
        root = 0
        while True:
            if root * root > x:
                return root - 1
            root += 1

# 70. Climbing Stairs
# You are climbing a staircase. It takes n steps to reach the top.
# Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?
# Example 1:
#     Input: n = 2
#     Output: 2
#     Explanation: There are two ways to climb to the top.
#     1. 1 step + 1 step
#     2. 2 steps
# Example 2:
#     Input: n = 3
#     Output: 3
#     Explanation: There are three ways to climb to the top.
#     1. 1 step + 1 step + 1 step
#     2. 1 step + 2 steps
#     3. 2 steps + 1 step
# 
    def climbStairs(self, n: int) -> int:
        pre = 0
        curr = 1
        for idx in range(n):
            curr = curr + pre
            pre = curr - pre
        return curr
        # recursive
        if n <= 2:
            return n
        else:
            return self.climbStairs(n-1) + self.climbStairs(n-2) 

# 88. Merge Sorted Array
# You are given two integer arrays nums1 and nums2, sorted in non-decreasing order, and two integers m and n, 
#   representing the number of elements in nums1 and nums2 respectively.
# Merge nums1 and nums2 into a single array sorted in non-decreasing order.
# The final sorted array should not be returned by the function, but instead be stored inside the array nums1. 
# To accommodate this, nums1 has a length of m + n, where the first m elements denote the elements that should be merged, 
#   and the last n elements are set to 0 and should be ignored. nums2 has a length of n.
#
# nums1.length == m + n
# nums2.length == n
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        for idx in range(n):
            nums1[m + idx] = nums2[idx]

        no_swap = True
        for idx in range(m + n - 1, 0 , -1):
            for bubble in range(idx):
                if nums1[bubble] > nums1[bubble + 1]:
                    nums1[bubble], nums1[bubble + 1] = nums1[bubble + 1], nums1[bubble]
                    no_swap = False
            if no_swap:
                break
        return 

    def merge_2(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        # add element from the tail, can do in one for loop
        nums1_end, nums2_end = m-1, n-1
        idx = m+n-1
        #for idx in range(m+n-1, 0 , -1):
        while(True):
            if nums1[nums1_end] > nums2[nums2_end]:
                nums1[idx] = nums1[nums1_end]
                nums1_end-=1
            else:
                nums1[idx] = nums1[nums2_end]
                nums2_end-=1
            if nums1_end >= 0 or nums2_end >=0:
                break
        return
    
#977. Squares of a Sorted Array
# Given an integer array nums sorted in non-decreasing order, 
#   return an array of the squares of each number sorted in non-decreasing order.
# Input: nums = [-4,-1,0,3,10]
# Output: [0,1,9,16,100]
# Explanation: After squaring, the array becomes [16,1,0,9,100].
# After sorting, it becomes [0,1,9,16,100].
    def sortedSquares(self, nums: List[int]) -> List[int]:
        left = 0
        right = len(nums) - 1
        opt = [None for num in nums]
        for idx in range(len(nums)):
            if abs(nums[left]) > abs(nums[right]):
                opt[idx] = nums[left] ** 2
                left += 1
            else:
                opt[idx] = nums[right] ** 2
                right -= 1
        return opt[::-1]

# 11. Container With Most Water - Medium
# Given n non-negative integers a1, a2, ..., an , where each represents a point at coordinate (i, ai). n vertical lines are drawn such that the two endpoints of the line i is at (i, ai) and (i
# Find two lines, which, together with the x-axis forms a container, such that the container contains the most water.
# Notice that you may not slant the container.
# Input: height = [1,8,6,2,5,4,8,3,7]
# Output: 49
# 
# P.S. O(N) solution: https://leetcode.com/problems/container-with-most-water/discuss/6100/Simple-and-clear-proofexplanation  
#
    def maxArea(self, height: List[int]) -> int:
        front, last = 0 , len(height) - 1
        maxarea = 0
        while front != last:
            compare = min(height[front], height[last]) * (last - front)
            maxarea = max(compare, maxarea)
            
            if height[front] > height[last]:
                last -= 1
            else:
                front += 1

        return maxarea

# 42. Trapping Rain Water - Hard
# Given n non-negative integers representing an elevation map where the width of each bar is 1, 
# compute how much water it can trap after raining.
#
    def trap(self, height: List[int]) -> int:

        return
 

 # 118. Pa#scal's Triangle
# Given an integer numRows, return the first numRows of Pascal's triangle.
# In Pascal's triangle, each number is the sum of the two numbers directly above it as shown:
    def generate(self, numRows: int) -> List[List[int]]:
        optList = []

        for idx in range(numRows):
            optList.append([1] * (idx + 1))
            if idx > 1:
                for j in range(1,idx):
                    optList[idx][j]=optList[idx-1][j-1]+optList[idx-1][j]
        return optList

# 121. Best Time to Buy and Sell Stock
# You are given an array prices where prices[i] is the price of a given stock on the ith day.
# You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.
# Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.
# [7,1,5,3,6,4]
    def maxProfit(self, prices: List[int]) -> int:
        opt = 0
        low_prices = prices[0]
        for idx in range(len(prices)):
            if prices[idx] < low_prices:
                low_prices = prices[idx]
            opt = max(opt, prices[idx] - low_prices)    
        return opt

    #20240804
    def maxProfit_20240804(self, prices: List[int]) -> int:
        # divede into:
        # 1. if 7 is buy, then find the biggest num in [1,5,3,6,4]
        max_val = prices[1]
        for i in range(1, len(prices)):
            if max_val < prices[i]:
                max_val = prices[i]
        option1 = max_val - prices[0]
        # 2. if not buying, find [1,5,3,6,4]
        option2 = self.maxProfit_20240804(prices[1:])
        return max(option1, option2)
    #result: Memory Limit Exceeded



# 125. Valid Palindrome
# A phrase is a palindrome if, after converting all uppercase letters into lowercase letters and removing all non-alphanumeric characters, it reads the same forward and backward. Alphanumeric characters include letters and numbers.
# Given a string s, return true if it is a palindrome, or false otherwise.  
    def isPalindrome(self, s: str) -> bool:
        opt = ''
        for ch in s.lower():
            if ch.isalnum():
                opt += ch
        return opt == opt[::-1]

# 136. Single Number ***
# Given a non-empty array of integers nums, every element appears twice except for one. Find that single one.
# You must implement a solution with a linear runtime complexity and use only constant extra space.
#
    def singleNumber(self, nums: List[int]) -> int:
        opt = 0  
        for num in nums:
            opt = opt ^ num     #using XOR
        return opt

# 167. Two Sum II - Input Array Is Sorted
# Given a 1-indexed array of integers numbers that is already sorted in non-decreasing order, find two numbers such that they add up to a specific target number. Let these two numbers be numbers[index1] and numbers[index2] where 1 <= index1 < index2 <= numbers.length.
# Return the indices of the two numbers, index1 and index2, added by one as an integer array [index1, index2] of length 2.
# The tests are generated such that there is exactly one solution. You may not use the same element twice.
    def twoSum167(self, numbers: List[int], target: int) -> List[int]:
        front = 0
        rear = len(numbers) - 1
        while front != rear:
            if numbers[front] + numbers[rear] > target:
                rear -= 1 
            elif numbers[front] + numbers[rear] < target:
                front += 1
            else:
                return [front + 1, rear + 1]
        
#168. Excel Sheet Column Title
# Given an integer columnNumber, return its corresponding column title as it appears in an Excel sheet.
# For example:
# A -> 1
# B -> 2
# C -> 3
# ...
# Z -> 26
# AA -> 27
# AB -> 28 
# ...
    def convertToTitle(self, columnNumber: int) -> str:
        strOut = ""
        mod = 0
        while columnNumber != 0:
            columnNumber -= 1
            mod = columnNumber % 26
            columnNumber = columnNumber // 26

            strOut = chr(mod + 65) + strOut
        
        return strOut

# 191. Number of 1 Bits
# Write a function that takes an unsigned integer and returns the number of '1' bits it has (also known as the Hamming weight).
# Constraints:
#   The input must be a binary string of length 32.    
    def hammingWeight(self, n: int) -> int:
        optNum = 0
        while n != 0:
            if n % 2 == 1:
                optNum += 1
            n = n // 2
        
        return optNum

# 190. Reverse Bits
# Reverse bits of a given 32 bits unsigned integer.
    def reverseBits(self, n: int) -> int:
        opt = 0
        for idx in range(32):
            opt = opt << 1 + n & 1  
            n = n >> 1
        return opt

# 217. Contains Duplicate
# Given an integer array nums, return true if any value appears at least twice in the array, and return false if every element is distinct.
    def containsDuplicate(self, nums: List[int]) -> bool:
        resList = []
        for num in nums:
            if num in resList:
                return True
            else:
                resList.append(num)        
        return False
    # using set, which can only store different elements
    def containsDuplicate2(self, nums: List[int]) -> bool:
        set1 = set(nums)       
        return not len(nums) == len(set1)

# 202. Happy Number
# Write an algorithm to determine if a number n is happy.
# A happy number is a number defined by the following process:
# Starting with any positive integer, replace the number by the sum of the squares of its digits.
# Repeat the process until the number equals 1 (where it will stay), or it loops endlessly in a cycle which does not include 1.
# Those numbers for which this process ends in 1 are happy.
# Return true if n is a happy number, and false if not.
    def isHappy(self, n: int) -> bool:
        loop_n = [n]
        while n != 1:
            list_n = [int(digit) for digit in str(n)]
            n = 0
            for ele in list_n:
                n += pow(ele, 2)
            
            if n in loop_n:
                return False
            else:
                loop_n.append(n)
        return True 

# 27. Remove Element
# Given an integer array nums and an integer val, remove all occurrences of val in nums in-place. The order of the elements may be changed. Then return the number of elements in nums which are not equal to val.
# Consider the number of elements in nums which are not equal to val be k, to get accepted, you need to do the following things:
# Change the array nums such that the first k elements of nums contain the elements which are not equal to val. The remaining elements of nums are not important as well as the size of nums.
    def removeElement(self, nums, val) -> int:
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        i = 0
        while i < len(nums):
            if val == nums[i]:
                nums.remove(val)
            else:
                i += 1
        return len(nums)
# Time complexity: O(N) 

#55. Jump Game
# You are given an integer array nums. You are initially positioned at the array's first index, 
# and each element in the array represents your maximum jump length at that position.
# Return true if you can reach the last index, or false otherwise.
# Example 1:
# Input: nums = [2,3,1,1,4]
# Output: true
# Explanation: Jump 1 step from index 0 to 1, then 3 steps to the last index.
#
# Example 2:
# Input: nums = [3,2,1,0,4]
# Output: false
# Explanation: You will always arrive at index 3 no matter what. Its maximum jump length is 0, which makes it impossible to reach the last index.
    def canJump(self, nums: List[int]) -> bool:
        jump_point = 0     
        for i in range(len(nums)):
            jump_point = max(jump_point, nums[i])
            if jump_point >= len(nums)-1-i:
                return True
            else:
                if jump_point == 0:
                    return False
                else:
                    jump_point -= 1
        return False
#   solution concept: 20240806
#                   check from nums[0~N], consider every number in array as your steps(like RPG game), and check the steps can jump to the end of the array.
#                   if steps not enough, move 1 step to next number, if the number is greater than your steps, replace the number as your new steps.
#   time complexity: O(N)
#   space complexity:O(1)


#45. Jump Game II
# You are given a 0-indexed array of integers nums of length n. You are initially positioned at nums[0].
# Each element nums[i] represents the maximum length of a forward jump from index i. In other words, if you are at nums[i], you can jump to any nums[i + j] where:
# 0 <= j <= nums[i] and
# i + j < n
# Return the minimum number of jumps to reach nums[n - 1]. The test cases are generated such that you can reach nums[n - 1].

# Example 1:
# Input: nums = [2,3,1,1,4]
# Output: 2
# Explanation: The minimum number of jumps to reach the last index is 2. Jump 1 step from index 0 to 1, then 3 steps to the last index.
#
# Example 2:
# Input: nums = [2,3,0,1,4]
# Output: 2

# Constraints:
# 1 <= nums.length <= 104
# 0 <= nums[i] <= 1000
# It's guaranteed that you can reach nums[n - 1].
#
# Intuition: Two pointer(slow/fast):   
#       站在position的位置，nums[position] = jump_point = 代表現在可以看到的視野，最遠可以看到 postion + nums[position]
#       視野 = fast = postion~postion+nums[postion]，往前尋找下一個適合的休息點(next_position)，站在那可以看到更遠的地方(nums[fast] > jump_point)。
#       視野(fast)往前看的同時，jump_point每次-1，若途中有更大的值就更新(nums[fast] > jump_point)
#       看到的位置已經到目前位置+nums[position]，就站到最大next_position的地點(position=next_position)，同時count+1
#       結束條件:如果目前站的位置已經可以跳到最後，則結束。站在position，視野=fast
    def jump(self, nums: List[int]) -> int:
        jump_point = 0
        position = 0
        min_count = 0
        for fast in range(len(nums)):
            # 發現目前的已經可以看到最後
            if position + nums[position] >= len(nums)-1:
                return min_count
            
            if nums[fast] > jump_point:
                jump_point = nums[fast]
                next_position = fast #可以看見視野內的最大值
            next_position = position + jump_point
            jump_point -= 1
            if fast == position + nums[position]:
                position = next_position
                min_count += 1
        return min_count
    
            

# 678. Valid Parenthesis String                
# Given a string s containing only three types of characters: '(', ')' and '*', return true if s is valid.
# The following rules define a valid string:
# Any left parenthesis '(' must have a corresponding right parenthesis ')'.
# Any right parenthesis ')' must have a corresponding left parenthesis '('.
# Left parenthesis '(' must go before the corresponding right parenthesis ')'.
# '*' could be treated as a single right parenthesis ')' or a single left parenthesis '(' or an empty string "".
class Solution:
    def checkValidString(self, s: str) -> bool:
        mincount = maxcount = 0
        for i in range(len(s)):
            if s[i] == '(':
                mincount += 1
                maxcount += 1
            elif s[i] == ')':
                mincount -= 1
                maxcount -= 1
            else:               # s[i] == '*'
                mincount -= 1   # take * as ')'
                maxcount += 1   # take * as '('
            
            # if not consider '*', count == 0 mean valid
            # count < 0 , means there are ')' occur before '('
            if maxcount < 0: 
                return False

            # if * occur when count == 0 , the '*' must be ' ' or '('        
            if mincount < 0:
                mincount = 0
        return mincount == 0
# 1029. Two City Scheduling
# A company is planning to interview 2n people. 
# Given the array costs where costs[i] = [aCosti, bCosti], 
#       the cost of flying the ith person to city a is aCosti, 
#       and the cost of flying the ith person to city b is bCosti.
# Return the minimum cost to fly every person to a city such that exactly n people arrive in each city.
    def key_func1(self, list_n):
        diff = list_n[2]
        return diff

    def twoCitySchedCost(self, costs: List[List[int]]) -> int:
        for i in range(len(costs)):
            costs[i].append(costs[i][0] - costs[i][1])
        
        costs.sort(key=self.key_func1)
        
        total_cost = 0
        for i in range(len(costs)):
            if i < len(costs)//2:
                #print(costs[i][0])
                total_cost += costs[i][0]
            else:
                #print(costs[i][1])
                total_cost += costs[i][1]
            #print(str(i) + ":" + str(total_cost))
        return total_cost
#精簡:  
    def twoCitySchedCost2(self, costs: List[List[int]]) -> int:
        #用costs[0] - costs[1]的結果來排序
        costs.sort(key = lambda x: x[0] - x[1])
        total_cost = 0
        for i in range(len(costs)):
            if i < len(costs)//2:
                total_cost += costs[0]
            else:
                total_cost += costs[1]
        return total_cost
        
     