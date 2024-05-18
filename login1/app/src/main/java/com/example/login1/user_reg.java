package com.example.login1;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import com.android.volley.AuthFailureError;
import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;

import org.json.JSONObject;

import java.util.HashMap;
import java.util.Map;

public class user_reg extends AppCompatActivity {

    Button b1;
    EditText ed1,ed2,ed3,ed4;
    TextView tv1;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_user_reg);
        ed1=(EditText)findViewById(R.id.username);
        ed2=(EditText)findViewById(R.id.password);

        ed3=(EditText)findViewById(R.id.email);
        ed4=(EditText)findViewById(R.id.phn);
        tv1=(TextView)findViewById(R.id.gotologin);
        tv1.setOnClickListener(new View.OnClickListener()
        {
            @Override
            public void onClick(View view)
            {
                Intent i2=new Intent(user_reg.this,MainActivity.class);
                startActivity(i2);
            }
        });
        b1=(Button)findViewById(R.id.regButton);
        b1.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View view) {
                Log.e("1","Register button Clicked --------");

                String username=ed1.getText().toString();
                String pass=ed2.getText().toString();

                String email=ed3.getText().toString();
                String phn=ed4.getText().toString();


                if (username.equals("")||pass.equals("")||pass.equals("")||email.equals("")||phn.equals("")){
                    Toast.makeText(getApplicationContext(),"Please provide full details",Toast.LENGTH_SHORT).show();
                }
                else if((username.length()<5)||(pass.length()<5))
                {
                    Toast.makeText(getApplicationContext(),"Username/Password should contain atleast 5 characters",Toast.LENGTH_SHORT).show();
                }
                else if (phn.length()==10){
                    Log.e("Entered here","Entered here");
                    RequestQueue requestQueue= Volley.newRequestQueue(getApplicationContext());
                    StringRequest requ=new StringRequest(Request.Method.POST, "http://192.168.42.214:8000/register/", new Response.Listener<String>() {
                        @Override
                        public void onResponse(String response) {

                            Log.e("Response is: ", response.toString());
                            try {
                                JSONObject o = new JSONObject(response);
                                String dat = o.getString("msg");
                                if(dat.equals("yes"))
                                {
                                    Toast.makeText(user_reg.this, "Registration Successful!", Toast.LENGTH_SHORT).show();
                                    Intent i1=new Intent(user_reg.this,MainActivity.class);
                                    startActivity(i1);
                                }
                                else if(dat.equals("Already registered"))
                                {
                                    Toast.makeText(user_reg.this, "This username is already taken", Toast.LENGTH_SHORT).show();
                                }
                                else
                                {
                                    Toast.makeText(user_reg.this, "Error Happened!!!", Toast.LENGTH_SHORT).show();
                                }
                            }
                            catch (Exception e){
                                e.printStackTrace();

                            }

                        }
                    }, new Response.ErrorListener() {
                        @Override
                        public void onErrorResponse(VolleyError error) {
//                Log.e(TAG,error.getMessage());
                            error.printStackTrace();
                        }
                    }){
                        @Override
                        protected Map<String, String> getParams() throws AuthFailureError {
                            Map<String,String> m=new HashMap<>();
                            m.put("username",username);
                            m.put("password",pass);
                            m.put("email",email);
                            m.put("phone",phn);


                            return m;
                        }
                    };
                    requestQueue.add(requ);
                }
                else
                {
                    Toast.makeText(getApplicationContext(),"Enter Valid Phone number",Toast.LENGTH_SHORT).show();
                }
            }
        });

    }
}