package com.example.login1;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.content.SharedPreferences;
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

public class MainActivity extends AppCompatActivity {
    EditText ed1;
    EditText ed2;
    Button loginButton;
    TextView tv1;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        ed1 =findViewById(R.id.username);
        ed2 = findViewById(R.id.password);
        tv1=(TextView)findViewById(R.id.gotoreg);
        tv1.setOnClickListener(new View.OnClickListener()
        {
            @Override
            public void onClick(View view)
            {
                Intent i2=new Intent(MainActivity.this,user_reg.class);
                startActivity(i2);
            }
        });
        loginButton = findViewById(R.id.loginButton);
        loginButton.setOnClickListener(new View.OnClickListener()
        {
            @Override
            public void onClick(View view)
            {
                String username=ed1.getText().toString();
                String password=ed2.getText().toString();

                if (username.equals("") || password.equals(""))
                {
                    Toast.makeText(MainActivity.this, "Please enter login details!!", Toast.LENGTH_SHORT).show();

                }
                else
                {
                    RequestQueue requestQueue= Volley.newRequestQueue(getApplicationContext());
                    StringRequest requ=new StringRequest(Request.Method.POST, "http://192.168.42.214:8000/find_login/", new Response.Listener<String>()
                    {
                        @Override
                        public void onResponse(String response) {

                            Log.e("Response is: ", response.toString());
                            try {
                                JSONObject o = new JSONObject(response);
                                String dat = o.getString("msg");
                                if(dat.equals("User")) {

                                    SharedPreferences sp = getSharedPreferences("userdetails", MODE_PRIVATE);
                                    SharedPreferences.Editor myEdit
                                            = sp.edit();
                                    myEdit.putString("username", username);
                                    myEdit.commit();

                                    Toast.makeText(MainActivity.this, "Login Successful!", Toast.LENGTH_SHORT).show();
                                    Intent i1 = new Intent(MainActivity.this, MainActivity2.class);
                                    startActivity(i1);
                                }
                                else
                                {
                                    Toast.makeText(MainActivity.this, "Invalid login details!", Toast.LENGTH_SHORT).show();

                                }

                            }
                            catch (Exception e){
                                e.printStackTrace();

                            }

                        }
                    }, new Response.ErrorListener()
                    {
                        @Override
                        public void onErrorResponse(VolleyError error)
                        {
//                Log.e(TAG,error.getMessage());
                            error.printStackTrace();
                        }
                    }){
                        @Override
                        protected Map<String, String> getParams() throws AuthFailureError
                        {
                            Map<String,String> m=new HashMap<>();
                            m.put("username",username);
                            m.put("password",password);

                            return m;
                        }
                    };
                    requestQueue.add(requ);
                }
            }
        });
    }
}