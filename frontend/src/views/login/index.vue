<template>
  <el-container>
    <el-header class="Header_Wrapper">
      <div class="NavBar_Wrapper">
        <el-col :span="12">
          <div class="Part1___Wrapper">
            <el-col :span="8" class="logoWrapper">
              <el-link href="/login" :underline="false">
                <img src="@/assets/logo.png" class="image_logo">
                Beta
              </el-link>
            </el-col>
            <el-col :span="8"></el-col>
            <el-col :span="8"></el-col>
          </div>
        </el-col>
        <el-col :span="12"></el-col>
      </div>
    </el-header>
    <el-main class="Main_Wrapper">
      <el-card class="box-card" shadow="hover">
        <div class="login-container">
          <el-form ref="loginForm" :model="loginForm" :rules="loginRules" class="login-form" auto-complete="on" label-position="left">

            <div class="title-container">
              <h3 class="style2">Login Form</h3>
            </div>

            <el-form-item prop="username">
        <span class="svg-container">
          <svg-icon icon-class="user" />
        </span>
              <el-input
                  ref="username"
                  v-model="loginForm.username"
                  placeholder="Username"
                  name="username"
                  type="text"
                  tabindex="1"
                  auto-complete="on"
              />
            </el-form-item>

            <el-form-item prop="password">
        <span class="svg-container">
          <svg-icon icon-class="password" />
        </span>
              <el-input
                  :key="passwordType"
                  ref="password"
                  v-model="loginForm.password"
                  :type="passwordType"
                  placeholder="Password"
                  name="password"
                  tabindex="2"
                  auto-complete="on"
                  @keyup.enter.native="handleLogin"
              />
              <span class="show-pwd" @click="showPwd">
          <svg-icon :icon-class="passwordType === 'password' ? 'eye' : 'eye-open'" />
        </span>
            </el-form-item>

            <el-button :loading="loading" type="primary" style="width:100%;margin-bottom:30px;" @click.native.prevent="handleLogin">Login</el-button>

            <div class="tips">
              <span style="margin-right:20px;">username: Your E-mail</span>
              <span> password: More than 6 digits</span>
            </div>

          </el-form>
        </div>
      </el-card>
    </el-main>
  </el-container>
</template>

<script>
import { validUsername } from '@/utils/validate'

export default {
  name: 'temp',
  data() {
    const validateUsername = (rule, value, callback) => {
      if (!validUsername(value)) {
        callback(new Error('Please enter the correct user name'))
      } else {
        callback()
      }
    }
    const validatePassword = (rule, value, callback) => {
      if (value.length < 6) {
        callback(new Error('The password can not be less than 6 digits'))
      } else {
        callback()
      }
    }
    return {
      loginForm: {
        username: '',
        password: ''
      },
      loginRules: {
        username: [{ required: true, trigger: 'blur', validator: validateUsername }],
        password: [{ required: true, trigger: 'blur', validator: validatePassword }]
      },
      loading: false,
      passwordType: 'password',
      redirect: undefined
    }
  },
  watch: {
    $route: {
      handler: function(route) {
        this.redirect = route.query && route.query.redirect
      },
      immediate: true
    }
  },
  methods: {
    showPwd() {
      if (this.passwordType === 'password') {
        this.passwordType = ''
      } else {
        this.passwordType = 'password'
      }
      this.$nextTick(() => {
        this.$refs.password.focus()
      })
    },
    handleLogin() {
      this.$refs.loginForm.validate(valid => {
        if (valid) {
          this.loading = true
          this.$store.dispatch('user/login', this.loginForm).then(() => {
            this.$router.push({ path: this.redirect || '/' })
            this.loading = false
          }).catch(() => {
            this.loading = false
          })
        } else {
          console.log('error submit!!')
          return false
        }
      })
    }
  }
}
</script>

<style scoped>
.Main_Wrapper{
  height: 650px;
  /*height: 100%;*/
  position:relative;
  overflow-y: auto;
  background: #eff2f2;

  display:flex;
  justify-content: center;
}
.NavBar_Wrapper{
  display:flex;
  text-align: center;
  justify-content:center;/*主轴上居中*/
  align-items:center;/*侧轴上居中*/
}
.Part1___Wrapper{
  margin-top: 20px;
  display:flex;
  text-align: center;
  justify-content:center;/*主轴上居中*/
  align-items:center;/*侧轴上居中*/
}
.logoWrapper{
  display:flex;
  text-align: center;
  justify-content:center;/*主轴上居中*/
  align-items:center;/*侧轴上居中*/
}
.image_logo{
  height: 30px;
  width: 30px;
  position: absolute;
  right: 30px;
  top: -40%;
}

.box-card {

  width: 500px;
  height: 70%;
  border-radius: 20px;
}
</style>
