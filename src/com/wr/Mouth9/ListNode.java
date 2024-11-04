package com.wr.Mouth9;

/**
 * ClassName: ListNode
 * Description:
 * date: 2024/9/9 9:23
 *
 * @author Wang
 * @since JDK 1.8
 */
public class ListNode {
    public int val;
      public ListNode next;
      ListNode() {}
      public ListNode(int val) {
          this.val = val;
      }
      ListNode(int val, ListNode next) {
          this.val = val;
          this.next = next;
      }
}
