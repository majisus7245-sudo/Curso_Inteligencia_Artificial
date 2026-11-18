import java.util.HashSet;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
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
            
            // estado -> menor profundidad con la que se ha alcanzado.
            // Se permite volver a un estado si ahora se llega por un camino mas corto;
            // asi la busqueda no descarta la ruta optima y la IDS sigue siendo optima.
            Map<String, Integer> visited = new HashMap<>();
            Pila<NodoPuzzle> pila = new Pila<>(2000000);

            NodoPuzzle current = root;
            visited.put(current.getState(), current.getDepth());
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
                        Integer profundidadPrevia = visited.get(childState);
                        if (profundidadPrevia == null || aux.getInfo().getDepth() < profundidadPrevia) {
                            visited.put(childState, aux.getInfo().getDepth());
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

    /*
     * Busqueda Bidireccional:
     * Se lanzan dos BFS al mismo tiempo, una desde el estado inicial y otra
     * desde el estado meta. En cada vuelta se expande un nivel completo de la
     * frontera mas pequena. Cuando un estado generado por un lado ya fue
     * visitado por el otro lado, las dos busquedas se encontraron y la
     * longitud del camino es la suma de las profundidades de ambos lados.
     */
    public void busquedaBidireccional() {
        long startTime = System.currentTimeMillis();
        int nodosProcesados = 0;
        int nodosGenerados = 2; // raiz inicial + raiz meta
        int maxSpace = 0;

        if (root.getState().compareTo(endState) == 0) {
            imprimirResultados("Bidireccional", 0, 0, 1, 1, startTime);
            return;
        }

        NodoPuzzle meta = new NodoPuzzle(endState, null);

        // estado -> nodo, para saber a que profundidad lo vio cada lado
        Map<String, NodoPuzzle> visitadosInicio = new HashMap<>();
        Map<String, NodoPuzzle> visitadosMeta = new HashMap<>();
        Cola<NodoPuzzle> colaInicio = new Cola<>(2000000);
        Cola<NodoPuzzle> colaMeta = new Cola<>(2000000);

        visitadosInicio.put(root.getState(), root);
        visitadosMeta.put(meta.getState(), meta);
        colaInicio.Insertar(root);
        colaMeta.Insertar(meta);

        while (!colaInicio.Vacia() && !colaMeta.Vacia()) {
            boolean desdeInicio = colaInicio.getLength() <= colaMeta.getLength();
            Cola<NodoPuzzle> cola = desdeInicio ? colaInicio : colaMeta;
            Map<String, NodoPuzzle> propios = desdeInicio ? visitadosInicio : visitadosMeta;
            Map<String, NodoPuzzle> otros = desdeInicio ? visitadosMeta : visitadosInicio;

            // Se expande el nivel completo y se guarda el mejor punto de encuentro
            int mejorProfundidad = Integer.MAX_VALUE;
            int nodosEnNivel = cola.getLength();

            for (int i = 0; i < nodosEnNivel && cola.Retirar(); i++) {
                NodoPuzzle current = cola.getDr();
                nodosProcesados++;

                Lista<NodoPuzzle> children = current.generateChildren();
                Nodo<NodoPuzzle> aux = children.getFrente();

                while (aux != null) {
                    NodoPuzzle child = aux.getInfo();
                    String childState = child.getState();

                    if (!propios.containsKey(childState)) {
                        propios.put(childState, child);
                        nodosGenerados++;

                        NodoPuzzle encuentro = otros.get(childState);
                        if (encuentro != null) {
                            int total = child.getDepth() + encuentro.getDepth();
                            if (total < mejorProfundidad) mejorProfundidad = total;
                        } else {
                            cola.Insertar(child);
                        }
                    }
                    aux = aux.getSig();
                }

                int currentSpace = visitadosInicio.size() + visitadosMeta.size()
                        + colaInicio.getLength() + colaMeta.getLength();
                if (currentSpace > maxSpace) maxSpace = currentSpace;
            }

            if (mejorProfundidad != Integer.MAX_VALUE) {
                imprimirResultados("Bidireccional", mejorProfundidad, nodosProcesados, nodosGenerados, maxSpace, startTime);
                System.out.println("Nodos visitados desde el inicio: " + visitadosInicio.size()
                        + " | desde la meta: " + visitadosMeta.size());
                return;
            }
        }

        System.out.println("\nBidireccional: no existe solucion para este estado inicial.");
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