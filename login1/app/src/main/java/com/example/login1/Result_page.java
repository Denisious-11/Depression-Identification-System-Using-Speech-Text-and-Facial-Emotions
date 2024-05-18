package com.example.login1;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

public class Result_page extends AppCompatActivity {
    TextView t1,t2,t3;
    Button b1;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_result_page);
        t1=(TextView)findViewById(R.id.t1);
        t2=(TextView)findViewById(R.id.t2);
        t3=(TextView)findViewById(R.id.t3);
        b1=(Button)findViewById(R.id.b1);
        String resp1=getIntent().getStringExtra("result1");
        String resp2=getIntent().getStringExtra("result2");
        String resp3=getIntent().getStringExtra("result3");

        t1.setText("");
        t2.setText("");
        t3.setText("");

        if (resp1.equals("Safe : Depression Not Detected"))
        {
            t1.setText(resp1);
            t2.setText("");
            t3.setText("");
        }
        else
        {
            t1.setText(resp1);
            t2.setText("Level : "+resp2);
            t3.setText("Recommendation : "+resp3);
        }

        b1.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                t1.setText("");
                t2.setText("");
                t3.setText("");
                Intent i1=new Intent(Result_page.this,MainActivity2.class);
                startActivity(i1);
            }
        });
    }
}