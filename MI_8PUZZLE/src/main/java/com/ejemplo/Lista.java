package com.ejemplo;

public class Lista <T> {
	
	private Nodo<T> frente;
	private Nodo<T> fin;
	private T       dr;
	private int     largo;
	
	public Lista() {
		frente=fin=null;
		largo=0;
	}
	
	public boolean InsertarFrente(T dato) {
		Nodo<T> nuevo;
		try {
			nuevo= new Nodo<T>(dato);
		} catch (OutOfMemoryError e) {
			return false;
		}
		largo++;
		// primer nodo de la lista
		if(frente==null) {
			frente=fin=nuevo;
			return true;
		}
		// insertalo al inicio
		nuevo.setSig(frente);
		frente=nuevo;
		
		return true;
	}
	public boolean InsertarFinal(T dato) {
		Nodo<T> nuevo;
		try {
			nuevo=new Nodo<T>(dato);
		} catch (OutOfMemoryError e) {
			return false;
		}
		largo++;
		// primer nodo de la lista
		if(frente==null) {
			frente=fin=nuevo;
			return true;
		}
		// insertalo al final
		fin.setSig(nuevo);
		fin=nuevo;
		return true;
	}
	public boolean InsertarOrdenado(T dato) {
		if(dato==null)
			return false;
		
		if(  frente==null || dato.toString().compareTo(frente.getInfo().toString())<=0) {
			return InsertarFrente(dato);
		}
		if( dato.toString().compareTo(fin.getInfo().toString())>=0) {
			return InsertarFinal(dato);
		}
		
		
		Nodo<T> nuevo;
		try {
			nuevo=new Nodo<T>(dato);
		} catch (OutOfMemoryError e) {
			return false;
		}	
		largo++;
		boolean ban = false;
		Nodo<T> ant=null;
		Nodo<T> aux=frente;
		String texto=dato.toString();
		while ( aux != null) {
			if(aux.getInfo().toString().compareTo(texto)>0 ) {
				ban=true;
				break;
			}
			ant=aux;
			aux=aux.getSig();
		}
		ant.setSig(nuevo);
		nuevo.setSig(aux);
		return true;
	}
	public boolean Borrar(int pos) {
		if(pos<1 || pos>largo)
			return false;
		largo--;
		Nodo<T> ant=null;
		if(frente==fin) {
			dr=frente.getInfo();
			frente=fin=null;
			return true;
		}
		Nodo<T> aux=frente;
		for(int i=1; i<pos;i++) {
			ant=aux;
			aux=aux.getSig();
		}
		if(ant==null) { // se borra el primer nodos
			dr=frente.getInfo();
			frente=frente.getSig();
			return true;
		}
		if (aux == fin) { // si es el ultimo nodo
			dr=fin.getInfo();
			ant.setSig(null);
			fin=ant;
			return true;
		}
		// entre dos nodos
		dr=aux.getInfo();
		ant.setSig(aux.getSig());
		return true;
	}
	public boolean Borrar(T dato) {
		if( dato==null)
			return false;
		String cadena = dato.toString();
		Nodo<T> aux = frente;
		int pos=0;
		boolean encontro=false;
		while ( aux != null) {
			pos++;
			if(aux.getInfo().toString().equalsIgnoreCase(cadena)) {
				encontro=true;
				break;
			}
			aux=aux.getSig();
		}
		if(encontro)
			return Borrar(pos);
		return false;
	}
	public boolean Buscar(T dato) {
		if(dato==null)
			return false;
		String cadena=dato.toString();
		Nodo<T> aux = frente;
		boolean encontro=false;
		while ( aux != null) {
			if(aux.getInfo().toString().equalsIgnoreCase(cadena)) {
				encontro=true;
				break;
			}
			aux=aux.getSig();
		}
		if(encontro)
			dr=aux.getInfo();
		return encontro;
				
	}
	public int length() {
		return largo;
	}
	public Nodo<T> getFrente() {
		return frente;
	}

	public Nodo<T> getFin() {
		return fin;
	}

	public T getDr() {
		return dr;
	}
	

}
