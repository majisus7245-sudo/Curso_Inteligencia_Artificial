# 8-Puzzle - Búsquedas no informadas

Solución del 8-puzzle con los algoritmos de búsqueda no informada:

1. Búsqueda en Anchura (BFS)
2. Búsqueda en Profundidad (DFS)
3. Costo Uniforme (UCS)
4. Iterativa (profundidad limitada iterativa, IDS)
5. Bidireccional

La tabla comparativa (resultados y complejidad algorítmica) está en [`tabla.txt`](tabla.txt).

## Archivos

| Archivo | Descripción |
|---|---|
| `App.java` | Punto de entrada: define el estado inicial/meta y ejecuta las búsquedas |
| `PuzzleTree.java` | Implementación de los 5 algoritmos de búsqueda |
| `NodoPuzzle.java` | Estado del tablero, padre, profundidad, costo y generación de sucesores |
| `Cola.java` | Cola (FIFO) usada por BFS y Bidireccional |
| `Pila.java` | Pila (LIFO) usada por DFS e IDS |
| `Lista.java`, `Nodo.java` | Lista enlazada para los sucesores |
| `Utils.java` | Imprime el tablero en formato 3x3 |

## Ejecutar

Desde esta carpeta:

```powershell
javac -encoding UTF-8 *.java
java App
```
