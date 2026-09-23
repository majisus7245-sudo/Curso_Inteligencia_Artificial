package com.ejemplo;

import java.io.Serializable;

public class Cola <T> implements Serializable {
    private int maximo, frente, fin;
	private T [] cola;
	private T dr;
	public Cola() {
		this(2000000);
	}
	public Cola(int maximo) {
		this.maximo=maximo;
		cola=(T[]) new Object[maximo];
		frente=fin=-1;
	}
	
	public boolean Insertar(T dato) {
		if(Llena())
			return false;
//		fin++;
//		cola[fin]=dato;
//		if(frente==-1)
//			frente=0;
//		return true;
		if(frente==-1)
			frente=fin=0;
		else 
			fin++;
		cola[fin]=dato;
		return true;
	}
	public boolean Retirar() {
		if(Vacia())
			return false;
		dr=cola[frente];
		cola[frente]=null;
		if(frente==fin)
			frente=fin=-1;
		else
			frente++;
		return true;
	}
	public int getLength() {
		if(Vacia())
			return 0;
		return fin-frente+1;
	}
	public boolean Llena() {
		return fin==maximo-1;
	}
	public boolean Vacia() {
		return fin==-1;
	}
	public int getMaximo() {
		return maximo;
	}
	public T getDr() {
		return dr;
	}
}
