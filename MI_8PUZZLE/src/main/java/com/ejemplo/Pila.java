package com.ejemplo;

public class Pila <T> {
	private T [] pila;
	private int tope, tamañoMaximo;
	private T dr;
	private String msg;
	public Pila() {
		this(2000000);
	}
	public Pila(int tamañoMaximo) {
		this.tamañoMaximo=tamañoMaximo;
		
		pila=(T[]) new Object[tamañoMaximo];
		tope=-1;
	}
	public boolean Insertar(T dato) {
		if(Llena()) {
			msg="Pila llena (verflow)";
			return false;
		}
		pila[++tope]=dato;
		msg="Inserción pexitosa";
		return true;
	}
	public boolean Retirar() {
		if(Vacia()) {
			msg="Pila vacia (underflow)";
			return false;
		}
		dr=pila[tope];
		pila[tope--]=null;
		msg="Retiro exitoso";
		return true;
	}
	public boolean Llena() {
		return tope==tamañoMaximo-1;
	}

	public int getLength() {
		return tope+1;
	}

	public boolean Vacia() {
		return tope==-1;
	}
	public T getDr() {
		return dr;
	}
	public String getMsg() {
		return msg;
	}
	public int getTamañoMaximo() {
		return tamañoMaximo;
	}
}
