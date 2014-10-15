package sistintel;

import java.io.*;
import java.util.Scanner;
import utilidades.LeeFicheroTXT;

public class ContVocCons {
	
	public String palabra;
	public int total_caracteres = 0;
	public int total_vocales = 0;
	public int total_consonates = 0;
	
	public ContVocCons(String cadena){
        this.palabra = cadena;
	}
	
	public void setTexto(String mensaje){
		this.palabra = mensaje;
	}
	
	public String getTexto(){
		return this.palabra;
	}
	
	public int getTotalVocales(){
		return this.total_vocales;
	}
	
	public int getTotalConsonantes(){
		return this.total_consonates;
	}
	
	public void contarVocalesConsonantes(){
		this.total_caracteres = palabra.length();
        for (int i = 0; i < total_caracteres; i++){
	        if ((palabra.charAt(i)=='a') || (palabra.charAt(i)=='e') || 
	        		(palabra.charAt(i)=='i') || (palabra.charAt(i)=='o')||
	                (palabra.charAt(i)=='u')){
	           this.total_vocales++; }
	    }
    	this.total_consonates = this.total_caracteres - this.total_vocales;
	}
	
	public void mostrarASCII(){
		byte[] bytes = {};
		try{
    		bytes = new String(this.palabra).getBytes("ISO-8859-1");
    	}catch(UnsupportedEncodingException e){
    		System.out.println("Conjunto de caracteres no soportados");
    	}
		
		for(int i=0; i<bytes.length; i++){
			System.out.println("valor: " +bytes[i]);
		}
	}
	
	public void filtrarCaractereEspeciales(){
		CaracterLatino automata = new CaracterLatino("resources/alfabeto.txt");
		this.palabra = automata.reemplazarCaracteres(this.palabra);
	}
	
	public class CaracterLatino {
		public String[][] alfabeto = new String[27][];
		
		public CaracterLatino(String directorio){
			LeeFicheroTXT recurso = new LeeFicheroTXT(directorio, "sin espacios");
			String[] contenido = recurso.getRenglones();
			for(int i=0; i<contenido.length; i++){
				this.alfabeto[i] = contenido[i].split("-");
			}
		}
		
		public String reemplazarCaracteres(String texto){
			texto = texto.toLowerCase();
			char[] caract_texto = texto.toCharArray();
			String[] pivote;
			
			for(int i=0; i<caract_texto.length; i++){
				
				for(int j=0; j<27; j++){
					pivote = this.alfabeto[j];
					if( pivote.length>1 ){
						pivote = pivote[1].split(",");
						
						for(int k=0; k<pivote.length; k++){
							if(pivote[k].charAt(0)==caract_texto[i]){
								caract_texto[i] = this.alfabeto[j][0].charAt(0);
							}
						}
					}
				}
			}
			
			String nvo_mensaje = new String(caract_texto);
			return nvo_mensaje;
		}
		
	}
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		Scanner teclado = new Scanner (System.in);
        System.out.println("Ingresa una palabra: ");
        String entrada = teclado.nextLine();
        
        ContVocCons analisis = new ContVocCons(entrada);
        analisis.setTexto(entrada);
        analisis.filtrarCaractereEspeciales();
        analisis.contarVocalesConsonantes();
        entrada = analisis.getTexto();
        String mensaje_salida = "La palabra " + entrada + " contiene " +analisis.getTotalVocales()+ 
        		" vocales; y también contiene: " +analisis.getTotalConsonantes()+ " consonantes";
        System.out.println(mensaje_salida);
        
        teclado.close();
	}
}