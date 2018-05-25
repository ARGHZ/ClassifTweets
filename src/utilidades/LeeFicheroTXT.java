package utilidades;

import java.io.*;
import java.util.StringTokenizer;

public class LeeFicheroTXT {
	protected File archivo = null;
	public FileReader fr = null;
	public BufferedReader br = null;
	
	public String[] renglones = new String[27];
	
	public LeeFicheroTXT(String directorio, String opcion){
		try{
			this.archivo = new File(directorio);
			this.fr = new FileReader(this.archivo);
			this.br = new BufferedReader(this.fr);
			
			String linea, cadena_sinespacios;
			StringTokenizer stTexto;
			int contador = 0;
			while( (linea=this.br.readLine())!=null ){
				
				switch(opcion){
					case "sin espacios":
						cadena_sinespacios = "";
						stTexto = new StringTokenizer(linea);
						while(stTexto.hasMoreElements()){
							cadena_sinespacios += stTexto.nextElement();
						}
						this.renglones[contador] = cadena_sinespacios;
						break;
						
					default:
						this.renglones[contador] = linea;
						break;
				}
				contador++;
			}
		}catch(Exception e){
			e.printStackTrace();
		}finally{
			try{
				if( null!=this.fr ){
					this.fr.close();
				}
			}catch(Exception e2){
				e2.printStackTrace();
			}
		}
	}
	
	public String[] getRenglones(){
		return this.renglones;
	}
}
