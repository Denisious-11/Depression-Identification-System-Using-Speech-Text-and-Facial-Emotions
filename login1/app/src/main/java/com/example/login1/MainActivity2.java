package com.example.login1;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.AppCompatButton;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.MediaPlayer;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.text.Editable;
import android.util.Base64;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;


import com.android.volley.AuthFailureError;
import com.android.volley.DefaultRetryPolicy;
import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;
import com.google.android.material.textfield.TextInputEditText;

import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class MainActivity2 extends AppCompatActivity {
    public static final int CAMERA_PREM_CODE = 101;
    public static final int CAMERA_REQUEST_CODE = 102;
    Button cam_button,pre;
    ImageView im_v;
    String image1;

    private Button rcrd;
    private MediaRecorder mRecorder;
    private MediaPlayer mPlayer;
    TextView tid;
    TextInputEditText tarea;
    private static final String LOG_TAG = "AudioRecording";
    private static String mFileName = null;
    public static final int REQUEST_AUDIO_PERMISSION_CODE = 1;

    @SuppressLint("MissingInflatedId")
    @Override
    protected void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main2);

        im_v = findViewById(R.id.img_view);
        cam_button = findViewById(R.id.cam_b1);
        cam_button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
//                    Toast.makeText(MainActivity2.this, "Camera!", Toast.LENGTH_SHORT).show();
                askCameraPermission();
            }
        });

        rcrd=(Button)findViewById(R.id.rcrd);
        tid=(TextView)findViewById(R.id.tid);
        tarea=(TextInputEditText)findViewById(R.id.tarea2);
//        pre=(Button)findViewById(R.id.pre_b3);

        mFileName = Environment.getExternalStorageDirectory().getAbsolutePath();
        Log.e("envname", mFileName);

        mFileName += "/AudioRecording.mp3";



        rcrd.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                String btntxt=rcrd.getText().toString().toLowerCase();
                if (btntxt.equals("record audio")){
                    tid.setText("");

                    mRecorder = new MediaRecorder();
                    mRecorder.setAudioSource(MediaRecorder.AudioSource.MIC);
                    mRecorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4);
                    mRecorder.setAudioEncoder(MediaRecorder.AudioEncoder.AMR_NB);
                    mRecorder.setOutputFile(mFileName);
                    try {
                        mRecorder.prepare();
                    } catch (IOException e) {
                        Log.e(LOG_TAG, "prepare() failed");
                    }
                    mRecorder.start();
                    Toast.makeText(getApplicationContext(), "Recording Started", Toast.LENGTH_LONG).show();
                    rcrd.setText("stop audio and PREDICT");
                }

                else {

                    try {

                        mRecorder.stop();
                        mRecorder.release();
                        mRecorder = null;
                        Toast.makeText(getApplicationContext(), "Recording Stopped", Toast.LENGTH_LONG).show();

                        String text_msg=tarea.getText().toString();

                        Log.e("Myfile name:",mFileName);
                        byte[] audioBytes;
                        try {

                            // Just to check file size.. Its is correct i-e; Not Zero
                            File audioFile = new File(mFileName);
                            long fileSize = audioFile.length();

                            ByteArrayOutputStream baos = new ByteArrayOutputStream();
                            FileInputStream fis = new FileInputStream(new File(mFileName));
                            byte[] buf = new byte[1024];
                            int n;
                            while (-1 != (n = fis.read(buf)))
                                baos.write(buf, 0, n);
                            audioBytes = baos.toByteArray();

                            // Here goes the Base64 string
                            String audioBase64 = Base64.encodeToString(audioBytes, Base64.DEFAULT);
                            Log.e("audioBase64",audioBase64+"");
//
                            RequestQueue requestQueue = Volley.newRequestQueue(getApplicationContext());
                            StringRequest requ = new StringRequest(Request.Method.POST, "http://192.168.29.46:8000/predict_depression/", new Response.Listener<String>() {
                                @Override
                                public void onResponse(String response) {

                                    Log.e("Response is: ", response.toString());
                                    try {
                                        JSONObject o = new JSONObject(response);
                                        String dat = o.getString("msg");

                                        if (dat.equals("yes")){
                                            String out1 = o.getString("out1");
                                            String out2 = o.getString("out2");
                                            String out3 = o.getString("out3");
                                            Intent i1=new Intent(MainActivity2.this,Result_page.class);
                                            i1.putExtra("result1",out1);
                                            i1.putExtra("result2",out2);
                                            i1.putExtra("result3",out3);
                                            startActivity(i1);
//                                            tid.setText(resp);
//                                            Toast.makeText(getApplicationContext(),resp,Toast.LENGTH_LONG).show();

                                        }
                                        else if(dat.equals("Depression Not Detected")){
                                            String out1 = o.getString("out1");
                                            Intent i1=new Intent(MainActivity2.this,Result_page.class);
                                            i1.putExtra("result1",out1);
                                            startActivity(i1);

                                        }
                                        else {

                                            Toast.makeText(getApplicationContext(),"Some error occured!!",Toast.LENGTH_LONG).show();
                                        }





                                    } catch (Exception e) {
                                        e.printStackTrace();

                                    }

                                }
                            }, new Response.ErrorListener() {
                                @Override
                                public void onErrorResponse(VolleyError error) {
//                Log.e(TAG,error.getMessage());
                                    error.printStackTrace();
                                }
                            }) {
                                @Override
                                protected Map<String, String> getParams() throws AuthFailureError {
                                    Map<String, String> m = new HashMap<>();

                                    m.put("audval", audioBase64);
                                    m.put("image",image1);
                                    m.put("text",text_msg);


                                    return m;
                                }
                            };
                            requ.setRetryPolicy(new DefaultRetryPolicy(
                                    60000,
                                    DefaultRetryPolicy.DEFAULT_MAX_RETRIES,
                                    DefaultRetryPolicy.DEFAULT_BACKOFF_MULT)
                            );
                            requestQueue.add(requ);



                        } catch (Exception e) {
                            Log.e("Error",e+"");
                        }


                        rcrd.setText("record audio");

                    }
                    catch (Exception e){

                        Log.e("Myfile name:",mFileName);
                        byte[] audioBytes;
                        try {
                            String text_msg=tarea.getText().toString();
                            // Just to check file size.. Its is correct i-e; Not Zero
                            File audioFile = new File(mFileName);
                            long fileSize = audioFile.length();

                            ByteArrayOutputStream baos = new ByteArrayOutputStream();
                            FileInputStream fis = new FileInputStream(new File(mFileName));
                            byte[] buf = new byte[1024];
                            int n;
                            while (-1 != (n = fis.read(buf)))
                                baos.write(buf, 0, n);
                            audioBytes = baos.toByteArray();

                            // Here goes the Base64 string
                            String audioBase64 = Base64.encodeToString(audioBytes, Base64.DEFAULT);
                            Log.e("audioBase64",audioBase64+"");
//
                            RequestQueue requestQueue = Volley.newRequestQueue(getApplicationContext());
                            StringRequest requ = new StringRequest(Request.Method.POST, "http://192.168.29.46:8000/predict_depression/", new Response.Listener<String>() {
                                @Override
                                public void onResponse(String response) {

                                    Log.e("Response is: ", response.toString());
                                    try {
                                        JSONObject o = new JSONObject(response);
                                        String dat = o.getString("msg");

                                        if (dat.equals("yes")){
                                            String out1 = o.getString("out1");
                                            String out2 = o.getString("out2");
                                            String out3 = o.getString("out3");
                                            Intent i1=new Intent(MainActivity2.this,Result_page.class);
                                            i1.putExtra("result1",out1);
                                            i1.putExtra("result2",out2);
                                            i1.putExtra("result3",out3);
                                            startActivity(i1);
//                                            tid.setText(resp);
//                                            Toast.makeText(getApplicationContext(),resp,Toast.LENGTH_LONG).show();

                                        }
                                        else if(dat.equals("Depression Not Detected")){
                                            String out1 = o.getString("out1");
                                            Intent i1=new Intent(MainActivity2.this,Result_page.class);
                                            i1.putExtra("result1",out1);
                                            startActivity(i1);

                                        }
                                        else {

                                            Toast.makeText(getApplicationContext(),"Some error occured!!",Toast.LENGTH_LONG).show();
                                        }






                                    } catch (Exception e) {
                                        e.printStackTrace();

                                    }

                                }
                            }, new Response.ErrorListener() {
                                @Override
                                public void onErrorResponse(VolleyError error) {
//                Log.e(TAG,error.getMessage());
                                    error.printStackTrace();
                                }
                            }) {
                                @Override
                                protected Map<String, String> getParams() throws AuthFailureError {
                                    Map<String, String> m = new HashMap<>();

                                    m.put("audval", audioBase64);
                                    m.put("image",image1);
                                    m.put("text",text_msg);


                                    return m;
                                }
                            };
                            requ.setRetryPolicy(new DefaultRetryPolicy(
                                    60000,
                                    DefaultRetryPolicy.DEFAULT_MAX_RETRIES,
                                    DefaultRetryPolicy.DEFAULT_BACKOFF_MULT)
                            );
                            requestQueue.add(requ);



                        } catch (Exception e1) {
                            Log.e("Error",e1+"");
                        }


                        rcrd.setText("record audio");

                    }

                }
            }
        });


    }

    private void askCameraPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,new String[] {Manifest.permission.CAMERA},CAMERA_PREM_CODE);
        }
        else{
            openCamera();
        }
    }

    private void openCamera() {
//        Toast.makeText(MainActivity2.this, "Camera!", Toast.LENGTH_SHORT).show();
        Intent camera = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        startActivityForResult(camera, CAMERA_REQUEST_CODE);
    }

    @SuppressLint("MissingSuperCall")
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if (requestCode == CAMERA_REQUEST_CODE) {
            Bitmap image= (Bitmap) data.getExtras().get("data");
            im_v.setImageBitmap(image);
            image1 = getStringImage(image);
        }
    }

    @SuppressLint("MissingSuperCall")
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if(requestCode==CAMERA_PREM_CODE){
            if(grantResults.length>0 && grantResults[0]==PackageManager.PERMISSION_GRANTED){
                openCamera();
            }
            else{
                Toast.makeText(MainActivity2.this, "Camera Permission is required to use camera", Toast.LENGTH_SHORT).show();
            }
        }
    }
    public String getStringImage(Bitmap bmp)
    {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        bmp.compress(Bitmap.CompressFormat.JPEG, 100, baos);
        byte[] imageBytes = baos.toByteArray();
        String encodedImage = Base64.encodeToString(imageBytes, Base64.DEFAULT);
        return encodedImage;

    }
}