package com.ejemplo;

public class App {
    public static void main(String[] args) {
        
        //String initialState = "123456 78";
        String initialState = "63548 217";
        //String initialState = "524 83167";
        String endState = "12345678 ";

        PuzzleTree searchTree = new PuzzleTree(initialState, endState);

        searchTree.BFS();
        searchTree.DFS();
        searchTree.costoUniforme();
        searchTree.busquedaIterativa();

        
        System.out.println("=========================================");
        System.out.println("End");
    }
    
}
