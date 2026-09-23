package Arboles;

public class ArbolBB <T>{

	private NodoABB<T> raiz,nuevo;
	private T          dr;
	
	public ArbolBB() {
		raiz=null;
	}
	
	public boolean Borrar(T dato) {
		dr=null;
		return Retirar(raiz,null,false,dato);
	}
	
	public void vaciar() {
		vaciar(raiz, null);
	}
	
	private void vaciar(NodoABB<T> nodo, NodoABB<T> ant) {
		if(nodo == null)
			return;
		if(nodo.getSubIzq() != null)
			vaciar(nodo.getSubIzq(), nodo);
		
		if(nodo.getSubDer() != null)
			vaciar(nodo.getSubDer(), nodo);
		
		Retirar(raiz, null, false, nodo.getInfo());
		vaciar(ant, null);
	}
	
	private boolean Retirar(NodoABB<T> root,NodoABB<T> anterior,boolean sentido, T dato) {
		if( root==null)
			return false;
		
//		Comparable datoNuevo = (Comparable) dato;
//		Comparable datoRoot= (Comparable) root.getInfo();
		String datoNuevo=dato.toString();
		String datoRoot=root.getInfo().toString();
		int res = datoNuevo.compareTo(datoRoot);
		if(res>0)
			return Retirar(root.getSubDer(),root,true,dato);
		if(res<0)
			return Retirar(root.getSubIzq(),root,false,dato);
		// Encontró al nodo, señaloadp por root
			dr=root.getInfo();

		
		if( root.getSubDer()!=null && root.getSubIzq()!= null) {
			
			NodoABB<T> aux = root.getSubDer();
					
			while ( aux.getSubIzq() != null) {
				aux=aux.getSubIzq();
			}

			root.setInfo(aux.getInfo());
			dato=aux.getInfo();
			
			return Retirar( root.getSubDer(),root,true,dato);
		}
		else { 
			if( root == raiz) {
				if( root.getSubIzq()==null && root.getSubDer()==null) {
					raiz=null;
				}else {
					raiz=root.getSubDer()==null?root.getSubIzq():root.getSubDer();
				}
				return true;
			}
			
			
			// 1 o 0 hijos.
			// llegue por el subIzq
			if(sentido) {
				anterior.setSubDer(root.getSubIzq()==null?root.getSubDer():root.getSubIzq());

			} else {
				
				anterior.setSubIzq(root.getSubIzq()==null?root.getSubDer():root.getSubIzq());
			}
		}
		return true;
	}

	public boolean Buscar(T dato) {
		return Buscar(dato,raiz);
	}
	
	private boolean Buscar(T dato, NodoABB<T> root) {
		if(root==null)
			return false;
		int res=dato.toString().compareTo(root.getInfo().toString());
		if(res==0) {
			dr=root.getInfo();
			return true;
		}
		if(res<0)
			return Buscar(dato,root.getSubIzq());
		return Buscar(dato,root.getSubDer());
	}
	

	public NodoABB<T> getRaiz() {
		return raiz;
	}
	public void setRaiz(NodoABB<T> raiz) {
		this.raiz = raiz;
	}
	public T getDr() {
		return dr;
	}
	public void setDr(T dr) {
		this.dr = dr;
	}
	
}
