package com.ejemplo;

public class Nodo <T>{
	private T Info;
	private Nodo<T> sig;

	public Nodo(T d){  
		Info=d;
		sig=null;
	}

	public T getInfo() {
		return Info;
	}

	public void setInfo(T info) {
		Info = info;
	}

	public Nodo<T> getSig() {
		return sig;
	}
	
	public void setSig(Nodo<T> sig) {
		this.sig = sig;
	}

}

