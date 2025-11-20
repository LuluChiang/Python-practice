from typing import Optional


class ListNode:
    def __init__(self, val=0, next = None):
        self.val = val
        self.next = next

class SingleLinklist:
    def __init__(self):
        self.head = ListNode(None)
        
    def addNode(self, data):
        cur = self.head
        if cur.val == None:
            self.head = ListNode(data)
        else:
            while cur.next != None:
                cur = cur.next
            cur.next = ListNode(data)
    
    def addNodes(self, *args):
        for arg in args:
            self.addNode(arg)
    
    def Printlist(self):
        cur = self.head
        list_val = [cur.val]
        list_next = [cur.next]
        while cur.next != None:
            cur = cur.next
            list_val.append(cur.val)
            list_next.append(cur.next)
        print(list_val)  
        # print(list_next)    



class Solution:
# 237. Delete Node in a Linked List
# Write a function to delete a node in a singly-linked list. 
# You will not be given access to the head of the list, instead you will be given access to the node to be deleted directly.
# It is guaranteed that the node to be deleted is not a tail node in the list.
# 2025/11/19
    def deleteNode2(self, node:ListNode):
        node.val = node.next.val
        node.next = node.next.next

    
    def deleteNode(self, node:ListNode):
        node.val = node.next.val
        node.next = node.next.next

# 83. Remove Duplicates from Sorted List
# Given the head of a sorted linked list, delete all duplicates such that each element appears only once. Return the linked list sorted as well.
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if head != None:
            cur = head
            while cur.next != None:
                if cur.val == cur.next.val:
                    cur.next = cur.next.next
                else:
                    cur = cur.next         
        return head 

# 203. Remove Linked List Elements 
# Given the head of a linked list and an integer val, remove all the nodes of the linked list that has Node.val == val, and return the new head.
#
# TODO: try this recursive solution again
    def removeElements(self, head: Optional[ListNode], val: int) -> Optional[ListNode]:
        if head == None: 
            return head
        head.next = self.removeElements(head.next, val)

        if head.val == val:
            return head.next
        else:
            return head

# 206. Reverse Linked List
# Given the head of a singly linked list, reverse the list, and return the reversed list.
# Follow up: A linked list can be reversed either iteratively or recursively. Could you implement both?
#
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # iterative
        pre = None
        curr = head
        revhead = head.next
        while revhead != None:
            curr.next = pre
            pre = curr
            curr = revhead
            revhead = revhead.next
        curr.next = pre
        return curr

        # recursive
        if head == None or head.next == None:
            return head
        else:
            rev_head = self.reverseList(head.next)
            rev_cur = rev_head
            while rev_cur.next != None:
                rev_cur = rev_cur.next
            head.next = None
            rev_cur.next = head
            return rev_head

# 2. Add Two Numbers - medium
# You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.
# You may assume the two numbers do not contain any leading zero, except the number 0 itself.
#
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        curr = sumhead = ListNode()
        carry = 0
        while l1 != None or l2 != None or carry: 
            curr.next = ListNode()    
            sum = 0
            if l1:
                sum += l1.val
                l1 = l1.next
            if l2:
                sum += l2.val
                l2 = l2.next
        
            currval = (sum + carry) % 10
            carry = (sum + carry) // 10

            curr.next = ListNode(currval)
            curr = curr.next
        
        return sumhead.next


# 141. Linked List Cycle
# Given head, the head of a linked list, determine if the linked list has a cycle in it.
# There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the next pointer. Internally, pos is used to denote the index of the node that tail's next pointer is connected to. Note that pos is not passed as a parameter.
# Return true if there is a cycle in the linked list. Otherwise, return false.
#
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        fast = head
        slow = head
        while fast != None and fast.next != None:
            fast = fast.next.next
            slow = slow.next
            if fast == slow: 
                return True        
        return False

# 160. Intersection of Two Linked Lists  -> ****
# Given the heads of two singly linked-lists headA and headB, 
# return the node at which the two lists intersect. 
# If the two linked lists have no intersection at all, return null.
#
# Main idea: L1 + L2 = L2 + L1, then the last node will hit if they intersect
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        if not headA or not headB:
            return None
        l1 = headA
        l2 = headB
        while l1 != l2:
            if l1 == None:
                l1 = headB
            else:
                l1 = l1.next
            if l2 == None:
                l2 = headA
            else:
                l2 = l2.next
        return l1 
