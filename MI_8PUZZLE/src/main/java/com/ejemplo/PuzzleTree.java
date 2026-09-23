package com.ejemplo;

import java.util.HashSet;
import java.util.Comparator;
import java.util.PriorityQueue;
import java.util.Set;

public class PuzzleTree {

    private String endState;
    private NodoPuzzle root;
    
    public PuzzleTree(String initialState, String endState){
        this.endState = endState;
        this.root = new NodoPuzzle(initialState, null);
    }

    public void BFS(){
        long startTime = System.currentTimeMillis();
        int nodosProcesados = 0;
        int nodosGenerados = 1; 
        int maxSpace = 0;
        
        Set<String> visited = new HashSet<>();
        Cola<NodoPuzzle> cola = new Cola<>(2000000); 

        NodoPuzzle current = root;
        visited.add(current.getState());
        cola.Insertar(current);

        while(!cola.Vacia() && cola.Retirar()){
            nodosProcesados++;
            current = cola.getDr();
            
            int currentSpace = visited.size() + cola.getLength();
            if(currentSpace > maxSpace) maxSpace = currentSpace;

            if(current.getState().compareTo(endState) == 0){
                imprimirResultados("Anchura (BFS)", current.getDepth(), nodosProcesados, nodosGenerados, maxSpace, startTime);
                break;
            }
            
            Lista<NodoPuzzle> children = current.generateChildren();
            Nodo<NodoPuzzle> aux = children.getFrente();
            
            while(aux != null){
                String childState = aux.getInfo().getState();
                if (!visited.contains(childState)){
                    visited.add(childState);
                    cola.Insertar(aux.getInfo());
                    nodosGenerados++;
                }
                aux = aux.getSig();
            }
        }
    }

    public void DFS(){
        long startTime = System.currentTimeMillis();
        int nodosProcesados = 0;
        int nodosGenerados = 1;
        int maxSpace = 0;
        
        Set<String> visited = new HashSet<>();
        Pila<NodoPuzzle> pila = new Pila<>(2000000);

        NodoPuzzle current = root;
        visited.add(current.getState());
        pila.Insertar(current);

        while(!pila.Vacia() && pila.Retirar()){
            nodosProcesados++;
            current = pila.getDr();
            
            int currentSpace = visited.size() + pila.getLength();
            if(currentSpace > maxSpace) maxSpace = currentSpace;

            if(current.getState().compareTo(endState) == 0){
                imprimirResultados("Profundidad (DFS)", current.getDepth(), nodosProcesados, nodosGenerados, maxSpace, startTime);
                break;
            }

            Lista<NodoPuzzle> children = current.generateChildren();
            Nodo<NodoPuzzle> aux = children.getFrente();
            
            while(aux != null){
                String childState = aux.getInfo().getState();
                if (!visited.contains(childState)){
                    pila.Insertar(aux.getInfo());
                    visited.add(childState);
                    nodosGenerados++;
                }
                aux = aux.getSig();
            }
        }
    }

    public void costoUniforme(){
        long startTime = System.currentTimeMillis();
        int nodosProcesados = 0;
        int nodosGenerados = 1;
        int maxSpace = 0;
        
        Set<String> visited = new HashSet<>();
        PriorityQueue<NodoPuzzle> queue = new PriorityQueue<>(Comparator.comparingInt(NodoPuzzle::getCost));

        NodoPuzzle current = root;
        visited.add(current.getState());
        queue.add(current);

        while(!queue.isEmpty()){
            current = queue.poll();
            nodosProcesados++;
            
            int currentSpace = visited.size() + queue.size();
            if(currentSpace > maxSpace) maxSpace = currentSpace;

            if(current.getState().compareTo(endState) == 0){
                imprimirResultados("Costo Uniforme (UCS)", current.getDepth(), nodosProcesados, nodosGenerados, maxSpace, startTime);
                System.out.println("Costo de la solución: " + current.getCost());
                break;
            }

            Lista<NodoPuzzle> children = current.generateChildren();
            Nodo<NodoPuzzle> aux = children.getFrente();
            
            while(aux != null){
                NodoPuzzle child = aux.getInfo();
                if (!visited.contains(child.getState())){
                    visited.add(child.getState());
                    queue.add(child);
                    nodosGenerados++;
                }
                aux = aux.getSig();
            }
        }
    }

    public void busquedaIterativa() {
        long startTime = System.currentTimeMillis();
        int maxDepth = 50;
        int totalProcesados = 0;
        int totalGenerados = 0;
        int maxSpace = 0;
        boolean metaEncontrada = false;

        for (int limit = 0; limit <= maxDepth; limit++) {
            
            Set<String> visited = new HashSet<>();
            Pila<NodoPuzzle> pila = new Pila<>(2000000);

            NodoPuzzle current = root;
            visited.add(current.getState());
            pila.Insertar(current);
            totalGenerados++; 

            while (!pila.Vacia() && pila.Retirar()) {
                totalProcesados++;
                current = pila.getDr();
                
                int currentSpace = visited.size() + pila.getLength();
                if(currentSpace > maxSpace) maxSpace = currentSpace;

                if (current.getState().compareTo(endState) == 0) {
                    imprimirResultados("Iterativa Limitada (IDS)", current.getDepth(), totalProcesados, totalGenerados, maxSpace, startTime);
                    System.out.println("Límite de profundidad donde se encontró: " + limit);
                    metaEncontrada = true;
                    break;
                }

                if (current.getDepth() < limit) {
                    Lista<NodoPuzzle> children = current.generateChildren();
                    Nodo<NodoPuzzle> aux = children.getFrente();
                    
                    while (aux != null) {
                        String childState = aux.getInfo().getState();
                        if (!visited.contains(childState)) {
                            visited.add(childState);
                            pila.Insertar(aux.getInfo());
                            totalGenerados++;
                        }
                        aux = aux.getSig();
                    }
                }
            }
            if (metaEncontrada) {
                break;
            }
        }
        
        if (!metaEncontrada) {
            System.out.println("No se encontró la meta dentro del límite de profundidad: " + maxDepth);
        }
    }

    private void imprimirResultados(String nombreAlgoritmo, int profundidad, int procesos, int nodos, int espacio, long startTime) {
        long endTime = System.currentTimeMillis();
        double segundos = (endTime - startTime) / 1000.0;
        
        System.out.println("\n--- RESULTADOS: " + nombreAlgoritmo + " ---");
        System.out.println("Meta encontrada en profundidad: " + profundidad);
        System.out.println("Nodos procesados (evaluados): " + procesos);
        System.out.println("Nodos generados (creados): " + nodos);
        System.out.println("Espacio máximo (nodos en memoria al mismo tiempo): " + espacio);
        System.out.println("Tiempo real: " + segundos + " segundos");
    }

    private void printPath(NodoPuzzle nodo) {
        Pila<NodoPuzzle> pathStack = new Pila<>(2000000);
        NodoPuzzle current = nodo;
        
        while (current != null) {
            pathStack.Insertar(current);
            current = current.getParent();
        }
        
        while(!pathStack.Vacia() && pathStack.Retirar()) {
            Utils.formatState(pathStack.getDr().getState());
        }
    }
}