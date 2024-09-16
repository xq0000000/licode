package com.wr;

import java.util.Scanner;

/**
 * ClassName: Soultion
 * Description:
 * date: 2024/8/18 19:33
 *
 * @author Wang
 * @since JDK 1.8
 */
class Solution {

    /* Write Code Here */
    public int numberOfPatrolBlocks(int[][] block) {
        int[][] a = {{1,0}, {0,-1}, {-1,0}, {0,1}};
        int sum = 0, k= 0;
        int m = block.length;
        int n = block[0].length;
        int i = 0;
        int flag = 0;
            for (int j = 0; j < n; ) {
                if (block[i][j] == 0) {
                    sum++;
                    if (k == 0) {
                        j++;
                        if (j>3) {
                            break;
                        }
                    }
                    if (k == 1) {
                        i++;
                        if (i>3) {
                            break;
                        }
                    }
                    if (k == 2) {
                        j--;
                        if (j<0) {
                            break;
                        }
                    }
                    if (k == 3) {
                        i--;
                        if (i<0) {
                            break;
                        }
                    }

                }
                if (i == m || j == n || block[i][j] == 1) {
                    i--;
                    j--;
                    if (k==3) {
                        k = 0;
                    }
                    if (i==m) {
                        i--;
                    }
                    if (j==n) {
                        j--;
                    }
                    i += a[k][0];
                    j += a[k][1];
                    if (block[i][j] == 1) {
                        break;
                    }
                    k++;
                }
//                if ((i == m-1 || j == n-1) && block[i][j] == 1) {
//                     break;
//                }
            }

        return sum;
    }
}

public class Main {
    public static void main(String[] args){
        Scanner in = new Scanner(System.in);
        int res;

        int block_rows = 0;
        int block_cols = 0;
        block_rows = in.nextInt();
        block_cols = in.nextInt();

        int[][] block = new int[block_rows][block_cols];
        for(int block_i=0; block_i<block_rows; block_i++) {
            for(int block_j=0; block_j<block_cols; block_j++) {
                block[block_i][block_j] = in.nextInt();
            }
        }

        if(in.hasNextLine()) {
            in.nextLine();
        }


        res = new Solution().numberOfPatrolBlocks(block);
        System.out.println(String.valueOf(res));

    }
}
