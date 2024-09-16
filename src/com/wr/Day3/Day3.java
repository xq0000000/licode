package com.wr.Day3;

public class Day3 {
    public static int[] finalPrices(int[] prices) {
        int j=0;
        for (int i = 0; i < prices.length; i++) {
            for (j = i+1; j < prices.length; j++) {
                if(prices[j]<=prices[i]){
                    break;
                }
            }
            if(j<prices.length){
                prices[i]=prices[i]-prices[j];
            }
        }
        return prices;
    }

    public static void main(String[] args) {
        int[] prices = new int[]{8,4,6,2,3};
        finalPrices(prices);
        for (int i = 0; i < prices.length; i++) {
            System.out.print(prices[i]);
        }
    }
}
